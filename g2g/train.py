"""Shared prior/grounder trainer with epoch-boundary resumable checkpoints."""

from __future__ import annotations

import csv
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from g2g.compose import ComposeConfig, detect_grasp
from g2g.data.dataset import GraspDataset, collate_samples
from g2g.data.grasp_maps import GraspMaps
from g2g.eval.grasp_rect import match_grasp
from g2g.losses import grounder_loss, prior_loss
from g2g.models.grconvnet3 import GraspPrior
from g2g.models.grounder import PartGrounder, parameter_counts
from g2g.utils.io import read_config, sha256_file, write_json
from g2g.utils.registry import append_registry, git_provenance, hardware_name, run_directory
from g2g.utils.seed import capture_rng, restore_rng, seed_everything


def build_model(config: dict, device: torch.device):
    if config["task"] == "prior":
        return GraspPrior(channel_size=config["channel_size"]).to(device), None
    if config["task"] != "grounder":
        raise ValueError("Only prior and grounder training are in the mandatory scope.")
    from transformers import CLIPSegProcessor

    model = PartGrounder.from_pretrained(config["model_id"], revision=config["model_revision"])
    revision = getattr(model.model.config, "_commit_hash", None) or config["model_revision"]
    config["model_revision"] = revision  # Record a resolved HF commit when available.
    processor = CLIPSegProcessor.from_pretrained(config["model_id"], revision=revision)
    return model.to(device), processor


def forward_batch(model, processor, batch: dict, device: torch.device, resolution: int):
    if processor is None:
        return model(batch["image"].to(device))
    inputs = processor(
        text=batch["text"],
        images=batch["raw_image"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    return model(inputs, output_size=(resolution, resolution))


@torch.no_grad()
def validate(model, processor, loader: DataLoader, device: torch.device, resolution: int) -> dict:
    model.eval()
    scores = []
    for batch in loader:
        output = forward_batch(model, processor, batch, device, resolution)
        if processor is not None:
            masks = output.sigmoid().cpu().numpy()[:, 0] > 0.5
            targets = batch["mask"].numpy()[:, 0] > 0.5
            for mask, target in zip(masks, targets, strict=True):
                union = (mask | target).sum()
                scores.append(float((mask & target).sum() / union) if union else 1.0)
        else:
            arrays = [tensor.float().cpu().numpy()[:, 0] for tensor in output]
            for index in range(len(batch["sample_id"])):
                maps = GraspMaps(*(array[index] for array in arrays))
                grasp = detect_grasp(maps, config=ComposeConfig(mode="prior", rerank_weight=0))
                if grasp is not None:
                    grasp = grasp.scaled(batch["original_size"][index] / resolution)
                scores.append(float(match_grasp(grasp, batch["union_rects"][index]).success))
    return {"val_miou" if processor is not None else "val_union_success": float(np.mean(scores))}


def train(
    config_path: str,
    manifest: str,
    val_manifest: str,
    data_root: str,
    results_root: str,
    run_id: str,
    *,
    resume: str | None = None,
    purpose: str = "debug",
    stop_after_epochs: int | None = None,
) -> dict:
    config = read_config(config_path)
    if purpose not in {"debug", "benchmark"}:
        raise ValueError("Purpose must be debug or benchmark.")
    for name in ("epochs", "batch_size", "resolution", "cpu_threads"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise ValueError("stop_after_epochs must be positive.")
    if config["learning_rate"] <= 0 or config["warmup_steps"] < 0 or config["grad_clip"] <= 0:
        raise ValueError("Invalid optimization settings.")
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = "prior" if config["task"] == "prior" else "instruction"
    training_data = GraspDataset(manifest, data_root, resolution=config["resolution"], task=task)
    validation_data = GraspDataset(
        val_manifest, data_root, resolution=config["resolution"], task=task
    )
    if training_data.rows[0]["split"] != "train" or validation_data.rows[0]["split"] != "val":
        raise ValueError("Training requires train.csv and a separate val split, never test rows.")
    if set(training_data.by_image) & set(validation_data.by_image):
        raise ValueError("Train/validation image leakage.")
    generator = torch.Generator().manual_seed(config["seed"])
    loader_options = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "collate_fn": collate_samples,
    }
    training_loader = DataLoader(training_data, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    directory = run_directory(results_root, run_id)
    shutil.copyfile(manifest, directory / "train_manifest.csv")
    shutil.copyfile(val_manifest, directory / "val_manifest.csv")
    provenance = git_provenance()
    started = time.perf_counter()
    hashes = {"train": sha256_file(manifest), "val": sha256_file(val_manifest)}
    resolved = {
        **config,
        "purpose": purpose,
        "manifest": str(Path(manifest).resolve()),
        "val_manifest": str(Path(val_manifest).resolve()),
        "manifest_hashes": hashes,
        "data_root": str(Path(data_root).resolve()),
        "resume": resume,
        "device": str(device),
        **provenance,
    }
    metrics = {"status": "failed", "purpose": purpose, "task": config["task"]}
    (directory / "config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    try:
        model, processor = build_model(config, device)
        resolved.update(config)
        (directory / "config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        total_steps = len(training_loader) * config["epochs"]
        warmup = min(config["warmup_steps"], max(0, total_steps // 10))

        def lr_ratio(step):
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (
                1
                + math.cos(
                    math.pi
                    * min(step - warmup, total_steps - warmup)
                    / max(1, total_steps - warmup)
                )
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_ratio)
        amp = bool(config["amp"] and device.type == "cuda")
        dtype = torch.bfloat16 if config["amp_dtype"] == "bf16" else torch.float16
        if amp and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
        start_epoch, best_score, global_step = 0, -math.inf, 0
        if resume:
            state = torch.load(resume, map_location="cpu", weights_only=True)
            if state["config"] != config or state["manifest_hashes"] != hashes:
                raise ValueError(
                    "Resume requires identical model/optimizer config and train/val manifests."
                )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_epoch, best_score, global_step = (
                state["epoch"] + 1,
                state["best_score"],
                state["global_step"],
            )
            generator.set_state(state["data_generator"])
            restore_rng(state["rng"])
            source_best = Path(resume).with_name("best.pt")
            if not source_best.exists():
                raise ValueError(
                    "Resume requires sibling best.pt to preserve best-checkpoint selection."
                )
            shutil.copyfile(source_best, directory / "best.pt")
        if start_epoch >= config["epochs"]:
            raise ValueError("Checkpoint has already completed this training budget.")
        end_epoch = config["epochs"]
        if stop_after_epochs is not None:
            end_epoch = min(end_epoch, start_epoch + stop_after_epochs)
        with (directory / "log.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["epoch", "train_loss", "val_score", "learning_rate"]
            )
            writer.writeheader()
            for epoch in range(start_epoch, end_epoch):
                model.train()
                total_loss, count = 0.0, 0
                for batch in training_loader:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                        output = forward_batch(
                            model, processor, batch, device, config["resolution"]
                        )
                        loss = (
                            prior_loss(output, tuple(t.to(device) for t in batch["maps"]))
                            if processor is None
                            else grounder_loss(output, batch["mask"].to(device))
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss.")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["grad_clip"], error_if_nonfinite=True
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1
                    size = len(batch["sample_id"])
                    total_loss += loss.item() * size
                    count += size
                validation = validate(
                    model, processor, validation_loader, device, config["resolution"]
                )
                score = next(iter(validation.values()))
                improved = score > best_score
                best_score = max(best_score, score)
                row = {
                    "epoch": epoch,
                    "train_loss": total_loss / count,
                    "val_score": score,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                writer.writerow(row)
                stream.flush()
                print(row, flush=True)
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "global_step": global_step,
                    "rng": capture_rng(),
                    "data_generator": generator.get_state(),
                    "config": config,
                    "manifest_hashes": hashes,
                }
                temporary = directory / "checkpoint.tmp"
                torch.save(state, temporary)
                temporary.replace(directory / "last.pt")
                if improved:
                    shutil.copyfile(directory / "last.pt", directory / "best.pt")
        metrics.update(
            status="completed" if end_epoch == config["epochs"] else "interrupted",
            epochs_completed=end_epoch,
            global_step=global_step,
            best_val_score=best_score,
            last_train_loss=total_loss / count,
            last_validation=validation,
            parameters=parameter_counts(model),
            effective_warmup_steps=warmup,
            amp_enabled=amp,
            amp_dtype=str(dtype) if amp else None,
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
                "kind": f"{purpose}_train_{config['task']}",
                **provenance,
                "config_path": str(directory / "config.yaml"),
                "manifest_sha256": hashes["train"],
                "seed": config["seed"],
                "hardware": hardware_name(),
                "wall_seconds": time.perf_counter() - started,
                "metrics_path": str(directory / "metrics.json"),
            },
        )
    return metrics
