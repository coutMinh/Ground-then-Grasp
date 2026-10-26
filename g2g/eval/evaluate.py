"""Top-1 native-resolution evaluation; reported-paper numbers never enter here."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from g2g.compose import ComposeConfig, detect_grasp
from g2g.data.dataset import GraspDataset, collate_samples
from g2g.data.grasp_maps import GraspMaps
from g2g.eval.grasp_rect import match_grasp
from g2g.train import build_model, forward_batch
from g2g.utils.io import sha256_file, write_json
from g2g.utils.registry import append_registry, git_provenance, hardware_name, run_directory

SYSTEMS = ("prior", "zero_shot_hard", "ft_hard", "soft", "ours", "mask_peak", "oracle_mask")


def load_expert(path: str, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state["config"]
    model, processor = build_model(config, device)
    model.load_state_dict(state["model"])
    return model.eval(), processor, config


def decomposition(records: list[dict]) -> dict:
    inside = [record for record in records if record["point_in_part"]]
    outside = [record for record in records if not record["point_in_part"]]
    return {
        "pointing_accuracy": len(inside) / len(records),
        "inside_count": len(inside),
        "outside_count": len(outside),
        "success_given_inside": sum(record["success"] for record in inside) / len(inside)
        if inside
        else None,
        "success_given_outside": sum(record["success"] for record in outside) / len(outside)
        if outside
        else None,
    }


@torch.no_grad()
def evaluate(
    manifest: str,
    data_root: str,
    prior_checkpoint: str,
    results_root: str,
    run_id: str,
    *,
    system: str = "prior",
    grounder_checkpoint: str | None = None,
    purpose: str = "debug",
    alpha: float = 0.5,
    temperature: float = 1,
    rerank_weight: float = 0.5,
    model_id: str = "CIDAS/clipseg-rd64-refined",
    model_revision: str = "main",
    compose_config: str | None = None,
) -> dict:
    if system not in SYSTEMS or purpose not in {"debug", "benchmark"}:
        raise ValueError("Unknown evaluation system or purpose.")
    if system in {"ft_hard", "soft", "ours", "mask_peak"} and not grounder_checkpoint:
        raise ValueError("This system requires a fine-tuned grounder checkpoint.")
    selection = None
    if compose_config:
        if system not in {"soft", "ours"}:
            raise ValueError("Frozen PoE config applies to soft/ours systems only.")
        selection = json.loads(Path(compose_config).read_text(encoding="utf-8"))
        if selection["selection_split"] != "val":
            raise ValueError("Composition must be selected on val.")
        if selection["prior_sha256"] != sha256_file(prior_checkpoint) or selection[
            "grounder_sha256"
        ] != sha256_file(grounder_checkpoint):
            raise ValueError("Frozen composition was selected for different checkpoints.")
        alpha = selection["composition"]["alpha"]
        temperature = selection["composition"]["temperature"]
        rerank_weight = selection["composition"]["rerank_weight"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior, _, prior_config = load_expert(prior_checkpoint, device)
    if prior_config["task"] != "prior":
        raise ValueError("Expected a prior checkpoint.")
    resolution = prior_config["resolution"]
    torch.set_num_threads(prior_config["cpu_threads"])
    dataset = GraspDataset(manifest, data_root, resolution=resolution)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_samples)
    split = dataset.rows[0]["split"]
    mode = {
        "prior": "prior",
        "zero_shot_hard": "hard",
        "ft_hard": "hard",
        "mask_peak": "mask_peak",
    }.get(system, "soft")
    composition = ComposeConfig(
        mode=mode,
        alpha=alpha,
        temperature=1 if system == "oracle_mask" else temperature,
        rerank_weight=rerank_weight if system in {"ours", "oracle_mask"} else 0,
    )
    directory = run_directory(results_root, run_id)
    shutil.copyfile(manifest, directory / "manifest.csv")
    started = time.perf_counter()
    provenance = git_provenance()
    config = {
        "system": system,
        "purpose": purpose,
        "split": split,
        "manifest": str(Path(manifest).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "prior_checkpoint": str(Path(prior_checkpoint).resolve()),
        "prior_sha256": sha256_file(prior_checkpoint),
        "grounder_checkpoint": grounder_checkpoint,
        "composition": asdict(composition),
        "resolution": resolution,
        "val_selection": selection,
        "iou_mode": "pixel",
        "iou_threshold": 0.25,
        "angle_boundary": "<=30 degrees",
        **provenance,
    }
    write_json(directory / "config.json", config)
    metrics = {"status": "failed", "purpose": purpose, "system": system, "split": split}
    try:
        grounder, processor = None, None
        if system not in {"prior", "oracle_mask"}:
            if system == "zero_shot_hard":
                grounder_config = {
                    "task": "grounder",
                    "model_id": model_id,
                    "model_revision": model_revision,
                }
                grounder, processor = build_model(grounder_config, device)
                grounder.eval()
                config["grounder_model"] = grounder_config
            else:
                grounder, processor, grounder_config = load_expert(grounder_checkpoint, device)
                if grounder_config["task"] != "grounder":
                    raise ValueError("Expected a grounder checkpoint.")
                config["grounder_sha256"] = sha256_file(grounder_checkpoint)
            write_json(directory / "config.json", config)
        records, mask_ious = [], []
        with (directory / "preds.jsonl").open("w", encoding="utf-8") as stream:
            for batch in loader:
                output = forward_batch(prior, None, batch, device, resolution)
                maps = GraspMaps(*(tensor[0, 0].float().cpu().numpy() for tensor in output))
                mask, logits = None, False
                if system == "oracle_mask":
                    mask = batch["mask"][0, 0].numpy()
                elif grounder is not None:
                    mask = (
                        forward_batch(grounder, processor, batch, device, resolution)[0, 0]
                        .float()
                        .cpu()
                        .numpy()
                    )
                    logits = True
                    binary = mask > 0
                    target = batch["mask"][0, 0].numpy() > 0.5
                    union = (binary | target).sum()
                    mask_ious.append(float((binary & target).sum() / union) if union else 1.0)
                grasp = detect_grasp(maps, mask, config=composition, mask_is_logits=logits)
                if grasp is not None:
                    grasp = grasp.scaled(batch["original_size"][0] / resolution)
                match = match_grasp(grasp, batch["gt_rects"][0])
                target_mask = batch["original_mask"][0]
                point_in_part = False
                if grasp is not None:
                    x, y = round(grasp.x), round(grasp.y)
                    point_in_part = bool(
                        0 <= y < target_mask.shape[0]
                        and 0 <= x < target_mask.shape[1]
                        and target_mask[y, x]
                    )
                record = {
                    "sample_id": batch["sample_id"][0],
                    "instruction": batch["text"][0],
                    "rectangle": grasp.to_dict() if grasp is not None else None,
                    **asdict(match),
                    "point_in_part": point_in_part,
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                records.append(record)
        metrics.update(
            status="completed",
            count=len(records),
            success_rate=sum(record["success"] for record in records) / len(records),
            grounding_miou=float(np.mean(mask_ious)) if mask_ious else None,
            decomposition=decomposition(records),
        )
    except Exception as error:
        metrics["error_type"] = type(error).__name__
        raise
    finally:
        write_json(directory / "metrics.json", metrics)
        append_registry(
            results_root,
            {
                "run_id": run_id,
                "kind": f"{purpose}_eval_{system}",
                **provenance,
                "config_path": str(directory / "config.json"),
                "manifest_sha256": sha256_file(manifest),
                "seed": prior_config["seed"],
                "hardware": hardware_name(),
                "wall_seconds": time.perf_counter() - started,
                "metrics_path": str(directory / "metrics.json"),
            },
        )
    return metrics
