"""Val-only PoE selection; each expert is forwarded once per sample."""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from g2g.compose import ComposeConfig, detect_grasp
from g2g.data.dataset import GraspDataset, collate_samples
from g2g.data.grasp_maps import GraspMaps
from g2g.data.manifest import read_manifest
from g2g.eval.evaluate import load_expert
from g2g.eval.grasp_rect import match_grasp
from g2g.train import forward_batch
from g2g.utils.io import sha256_file, write_json
from g2g.utils.registry import append_registry, git_provenance, hardware_name, run_directory


@torch.no_grad()
def sweep(
    manifest: str,
    data_root: str,
    prior_checkpoint: str,
    grounder_checkpoint: str,
    results_root: str,
    run_id: str,
    *,
    purpose: str = "debug",
) -> dict:
    rows = read_manifest(manifest)
    if not rows or {row["split"] for row in rows} != {"val"}:
        raise ValueError("Hyperparameter sweeps accept val samples only; never tune on test.")
    if purpose not in {"debug", "benchmark"}:
        raise ValueError("Unknown run purpose.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior, _, config = load_expert(prior_checkpoint, device)
    grounder, processor, grounder_config = load_expert(grounder_checkpoint, device)
    if config["task"] != "prior" or grounder_config["task"] != "grounder":
        raise ValueError("Sweep requires one prior and one fine-tuned grounder.")
    torch.set_num_threads(config["cpu_threads"])
    resolution = config["resolution"]
    dataset = GraspDataset(manifest, data_root, resolution=resolution)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_samples)
    candidates = [
        ComposeConfig(alpha=alpha, temperature=temperature, rerank_weight=weight)
        for alpha, temperature, weight in itertools.product(
            (0, 0.25, 0.5, 0.75, 1), (0.5, 1, 2), (0, 0.5, 1)
        )
    ]
    successes = [0] * len(candidates)
    directory = run_directory(results_root, run_id)
    provenance = git_provenance()
    experiment = {
        "purpose": purpose,
        "selection_split": "val",
        "manifest_sha256": sha256_file(manifest),
        "prior_sha256": sha256_file(prior_checkpoint),
        "grounder_sha256": sha256_file(grounder_checkpoint),
        "tie_rule": "first grid entry with maximum success rate",
        **provenance,
    }
    write_json(directory / "config.json", experiment)
    started = time.perf_counter()
    metrics = {"status": "failed", "purpose": purpose}
    try:
        for batch in loader:
            output = forward_batch(prior, None, batch, device, resolution)
            maps = GraspMaps(*(tensor[0, 0].float().cpu().numpy() for tensor in output))
            logits = (
                forward_batch(grounder, processor, batch, device, resolution)[0, 0]
                .float()
                .cpu()
                .numpy()
            )
            for index, candidate in enumerate(candidates):
                grasp = detect_grasp(maps, logits, config=candidate, mask_is_logits=True)
                if grasp is not None:
                    grasp = grasp.scaled(batch["original_size"][0] / resolution)
                successes[index] += match_grasp(grasp, batch["gt_rects"][0]).success
        best_index = max(range(len(candidates)), key=lambda index: successes[index])
        selected = {**experiment, "composition": asdict(candidates[best_index])}
        write_json(directory / "best_compose.json", selected)
        metrics.update(
            status="completed",
            count=len(dataset),
            best=selected,
            candidates=[
                {"composition": asdict(candidate), "success_rate": success / len(dataset)}
                for candidate, success in zip(candidates, successes, strict=True)
            ],
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
                "kind": f"{purpose}_val_sweep",
                **provenance,
                "config_path": str(directory / "config.json"),
                "manifest_sha256": experiment["manifest_sha256"],
                "seed": config["seed"],
                "hardware": hardware_name(),
                "wall_seconds": time.perf_counter() - started,
                "metrics_path": str(directory / "metrics.json"),
            },
        )
    return {"status": metrics["status"], "best_compose": str(directory / "best_compose.json")}
