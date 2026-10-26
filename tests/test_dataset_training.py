"""Synthetic native-format fixtures test behavior, never benchmark accuracy."""

import csv
import hashlib
import pickle

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from g2g.data.dataset import GraspDataset
from g2g.data.manifest import FIELDS, read_manifest, write_manifest
from g2g.data.schema import SampleKey
from g2g.eval.evaluate import evaluate
from g2g.train import train


@pytest.fixture
def native_fixture(tmp_path):
    root = tmp_path / "data"
    splits = {"train": [], "val": []}
    for number in range(5):
        image_id = hashlib.sha256(f"synthetic-{number}".encode()).hexdigest()
        split = "train" if number < 4 else "val"
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[10:22, 8:24] = (80, 160, 220)
        for part in (0, 1):
            key = SampleKey(image_id, 0, part)
            for member in key.members().values():
                (root / member).parent.mkdir(parents=True, exist_ok=True)
            members = key.members()
            Image.fromarray(image).save(root / members["image"])
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[10:22, 8:24] = 1
            np.save(root / members["mask"], mask)
            (root / members["instruction"]).write_bytes(pickle.dumps("Grasp the blue block."))
            torch.save(
                torch.tensor([[0.5, 14.0 + part * 4, 16, 12, 6, 0]]), root / members["label"]
            )
            splits[split].append(
                {
                    "sample_id": key.sample_id,
                    "image_id": image_id,
                    "object_id": key.object_id,
                    "split": split,
                    **{f"{field}_member": value for field, value in members.items()},
                }
            )
    for split, rows in splits.items():
        write_manifest(tmp_path / f"{split}.csv", rows)
    config = {
        "task": "prior",
        "seed": 42,
        "resolution": 32,
        "channel_size": 4,
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "warmup_steps": 2,
        "grad_clip": 1.0,
        "num_workers": 0,
        "amp": False,
        "amp_dtype": "bf16",
        "cpu_threads": 2,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return root, tmp_path / "train.csv", tmp_path / "val.csv", config_path


def test_prior_is_sampled_once_per_image_and_union_contains_both_parts(native_fixture):
    root, manifest, _, _ = native_fixture
    assert b"\r\n" not in manifest.read_bytes()  # Stable SHA-256 on Windows and molab/Linux.
    prior = GraspDataset(manifest, root, resolution=32, task="prior")
    instruction = GraspDataset(manifest, root, resolution=32, task="instruction")
    assert len(prior) == 4
    assert len(instruction) == 8
    assert len(prior[0]["union_rects"]) == 2
    assert len(prior[0]["gt_rects"]) == 1
    assert prior[0]["image"].shape == (3, 32, 32)


def test_combined_manifest_rejects_image_leakage(native_fixture, tmp_path):
    _, manifest, _, _ = native_fixture
    rows = read_manifest(manifest)
    rows[1]["split"] = "val"
    path = tmp_path / "leaky.csv"
    write_manifest(path, rows)
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(path)


def test_resume_reproduces_uninterrupted_training_and_evaluation(native_fixture, tmp_path):
    root, manifest, val_manifest, config = native_fixture
    results = tmp_path / "results"
    arguments = [str(config), str(manifest), str(val_manifest), str(root), str(results)]
    full = train(*arguments, "full")
    partial = train(*arguments, "partial", stop_after_epochs=1)
    assert partial["status"] == "interrupted"
    resumed = train(*arguments, "resumed", resume=str(results / "runs/partial/last.pt"))
    assert resumed["global_step"] == full["global_step"]
    first = torch.load(results / "runs/full/last.pt", weights_only=True)
    second = torch.load(results / "runs/resumed/last.pt", weights_only=True)
    for name in first["model"]:
        torch.testing.assert_close(first["model"][name], second["model"][name], rtol=0, atol=0)
    metrics = evaluate(
        str(val_manifest),
        str(root),
        str(results / "runs/full/best.pt"),
        str(results),
        "eval",
        system="prior",
    )
    assert metrics["count"] == 2
    assert metrics["status"] == "completed"
    union_metrics = evaluate(
        str(manifest),
        str(root),
        str(results / "runs/full/best.pt"),
        str(results),
        "union_eval",
        system="prior",
        union_targets=True,
    )
    assert union_metrics["count"] == 4
    assert union_metrics["decomposition"] is None
    with (results / "registry.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 5


def test_multi_digit_part_ids_and_no_path_traversal():
    key = SampleKey.parse("a" * 64 + "_12_34")
    assert key.object_index == 12 and key.part_index == 34
    with pytest.raises(ValueError):
        SampleKey.parse("../" + key.sample_id)
    assert len(FIELDS) == 8
