"""Gallery tests use synthetic fixtures; they never claim model performance."""

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from PIL import Image

from g2g.data.manifest import write_manifest
from g2g.data.schema import SampleKey
from g2g.eval.grasp_rect import GraspRectangle, match_grasp
from g2g.viz.predictions import select_samples, visualize_predictions


@pytest.fixture
def saved_evaluation(tmp_path):
    data_root = tmp_path / "data"
    run_dir = tmp_path / "results/runs/ours-full-unseen-01"
    run_dir.mkdir(parents=True)
    rows, predictions = [], []
    target = GraspRectangle(16, 16, 12, 6, 0)
    for index in range(2):
        key = SampleKey(f"{index + 1:064x}", 0, 0)
        members = key.members()
        for member in members.values():
            (data_root / member).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (80, 130, 190)).save(data_root / members["image"])
        torch.save(torch.tensor([[0.5, 16, 16, 12, 6, 0.0]]), data_root / members["label"])
        instruction = f"Grasp synthetic block {index}."
        (data_root / members["instruction"]).write_bytes(pickle.dumps(instruction))
        prediction = target if index == 0 else GraspRectangle(27, 27, 4, 2, 0)
        predictions.append(
            {
                "sample_id": key.sample_id,
                "instruction": instruction,
                "rectangle": prediction.to_dict(),
                **asdict(match_grasp(prediction, [target])),
                "point_in_part": index == 0,
            }
        )
        rows.append(
            {
                "sample_id": key.sample_id,
                "image_id": key.image_id,
                "object_id": key.object_id,
                "split": "test_unseen",
                **{f"{name}_member": value for name, value in members.items()},
            }
        )
    write_manifest(run_dir / "manifest.csv", rows)
    (run_dir / "preds.jsonl").write_text(
        "\n".join(json.dumps(record) for record in predictions) + "\n", encoding="utf-8"
    )
    for name, value in {
        "config": {"system": "ours", "target_scope": "instruction", "iou_mode": "pixel"},
        "metrics": {
            "status": "completed",
            "system": "ours",
            "split": "test_unseen",
            "count": 2,
            "success_rate": 0.5,
        },
    }.items():
        (run_dir / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    return run_dir, data_root


def test_gallery_renders_both_outcomes_without_changing_evaluation(saved_evaluation, tmp_path):
    run_dir, data_root = saved_evaluation
    original = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    report = visualize_predictions(run_dir, data_root, tmp_path / "gallery", limit=6)
    assert report["count"] == 2
    assert {sample["success"] for sample in report["samples"]} == {True, False}
    assert all(sample["matches_saved_metric"] for sample in report["samples"])
    with Image.open(report["contact_sheet"]) as image:
        assert image.width == 960 and image.height > 640
    for sample in report["samples"]:
        assert sample["gt_count"] == 1 and Path(sample["figure"]).is_file()
    assert original == {path.name: path.read_bytes() for path in run_dir.iterdir()}


def test_gallery_rejects_mismatched_manifest(saved_evaluation, tmp_path):
    run_dir, data_root = saved_evaluation
    path = run_dir / "preds.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["sample_id"] = "not-this-manifest"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="different samples"):
        visualize_predictions(run_dir, data_root, tmp_path / "gallery")


def test_sample_selection_is_reproducible_and_includes_failures():
    records = [{"sample_id": str(i), "success": i < 8} for i in range(12)]
    selected = select_samples(records, 6, "mixed", 42)
    assert selected == select_samples(list(reversed(records)), 6, "mixed", 42)
    assert sum(row["success"] for row in selected) == 3
    assert len(select_samples(records, 20, "mixed", 42)) == 12
