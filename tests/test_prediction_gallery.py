"""Gallery tests use synthetic fixtures; they never claim model performance."""

import ast
import json
import pickle
import runpy
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from g2g.data.manifest import write_manifest
from g2g.data.schema import SampleKey
from g2g.eval.grasp_rect import GraspRectangle, match_grasp
from g2g.viz.predictions import select_samples, visualize_predictions


@pytest.fixture
def gallery_sources():
    # A standalone Git checkout does not include the user's external notebooks.
    # Exercise the canonical generator in this repo without calling its file writer.
    builder = Path(__file__).resolve().parents[1] / "scripts/build_prediction_cell.py"
    namespace = runpy.run_path(str(builder))
    return namespace["TEMPLATE"], namespace["render_gallery_cells"]()


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


def test_pasteable_cell_and_notebook_call_repository_cli(
    saved_evaluation, tmp_path, gallery_sources
):
    import marimo as mo

    run_dir, data_root = saved_evaluation
    repo = Path(__file__).resolve().parents[1]
    source, notebook_source = gallery_sources
    assert "_renderer_source" not in source
    assert "ImageDraw" not in source
    commands = []

    def run(args, stage, cwd):
        commands.append([str(arg) for arg in args])
        return subprocess.run(commands[-1], cwd=cwd, check=True, capture_output=True)

    runtime = SimpleNamespace(
        artifacts=tmp_path / "artifacts",
        results=run_dir.parents[1],
        data_root=data_root,
        checkout=repo,
        require_setup=lambda: None,
        identifier=lambda value: value,
        unique_id=lambda prefix: "gallery",
        run=run,
    )
    exec(compile(source, "gallery-cell", "exec"), {"runtime": runtime, "mo": mo})
    report = json.loads((runtime.artifacts / "qualitative/gallery/selection.json").read_text())
    assert report["count"] == 2 and Path(report["contact_sheet"]).is_file()
    assert commands[0][1:4] == ["-m", "g2g", "visualize-predictions"]
    assert not (runtime.artifacts / "qualitative/_render_predictions.py").exists()
    # The generated app cell uses exactly the same CLI orchestration.
    notebook = ast.parse(notebook_source)
    gallery_cell = next(
        node
        for node in notebook.body
        if isinstance(node, ast.FunctionDef)
        and "gallery_form" in {arg.arg for arg in node.args.args}
    )
    gallery_cell.decorator_list = []
    namespace = {}
    exec(
        compile(ast.Module(body=[gallery_cell], type_ignores=[]), "gallery-cell", "exec"), namespace
    )
    runtime.unique_id = lambda prefix: "notebook-gallery"
    namespace["_"](
        gallery_form=SimpleNamespace(
            value={
                "run_id": run_dir.name,
                "limit": 2,
                "selection": "mixed",
                "seed": 42,
                "update_checkout": False,
                "repo_ref": "main",
            }
        ),
        mo=mo,
        runtime=runtime,
    )
    assert (runtime.artifacts / "qualitative/notebook-gallery/contact_sheet.png").is_file()


@pytest.fixture
def checkout_update(saved_evaluation, tmp_path, gallery_sources):
    def git(*args):
        return subprocess.check_output(
            ["git", *(str(arg) for arg in args)], stderr=subprocess.STDOUT, text=True
        ).strip()

    origin, seed, checkout = (tmp_path / name for name in ("origin.git", "seed", "checkout"))
    git("init", "--bare", str(origin))
    git("init", "--initial-branch=main", str(seed))
    git("-C", seed, "config", "user.email", "fixture@example.invalid")
    git("-C", seed, "config", "user.name", "Gallery fixture")
    (seed / "README.md").write_text("original\n", encoding="utf-8")
    git("-C", seed, "add", "README.md")
    git("-C", seed, "commit", "-m", "Initial fixture")
    old_commit = git("-C", seed, "rev-parse", "HEAD")
    git("-C", seed, "remote", "add", "origin", origin)
    git("-C", seed, "push", "origin", "main")
    git("clone", "--branch", "main", origin, checkout)
    module = seed / "g2g/viz/predictions.py"
    module.parent.mkdir(parents=True)
    module.write_text('"""Checkout-update fixture."""\n', encoding="utf-8")
    git("-C", seed, "add", "g2g/viz/predictions.py")
    git("-C", seed, "commit", "-m", "Add visualization fixture")
    git("-C", seed, "push", "origin", "main")
    new_commit = git("-C", seed, "rev-parse", "HEAD")
    run_dir, data_root = saved_evaluation
    repo = Path(__file__).resolve().parents[1]
    source = gallery_sources[0]
    source = source.replace("_update_checkout = False", "_update_checkout = True", 1)
    commands = []

    def run(args, stage, cwd):
        command = [str(arg) for arg in args]
        commands.append(command)
        # Exercise Git against the local remote; render with the actual package under test.
        return subprocess.run(
            command,
            cwd=repo if "visualize-predictions" in command else cwd,
            check=True,
            capture_output=True,
        )

    runtime = SimpleNamespace(
        artifacts=tmp_path / "artifacts",
        results=run_dir.parents[1],
        data_root=data_root,
        checkout=checkout,
        require_setup=lambda: None,
        identifier=lambda value: value,
        unique_id=lambda prefix: "gallery",
        run=run,
    )
    return runtime, source, git, old_commit, new_commit, commands


def test_checkout_update_fetches_published_module_and_keeps_results(checkout_update):
    import marimo as mo

    runtime, source, git, old_commit, new_commit, commands = checkout_update
    registry = runtime.results / "registry.csv"
    registry.write_text("preserved-results\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in runtime.results.rglob("*") if path.is_file()}
    assert git("-C", runtime.checkout, "rev-parse", "HEAD") == old_commit
    exec(compile(source, "gallery-cell", "exec"), {"runtime": runtime, "mo": mo})
    assert git("-C", runtime.checkout, "rev-parse", "HEAD") == new_commit
    assert commands[0][-3:] == ["fetch", "origin", "main"]
    assert commands[1][-3:] == ["checkout", "--detach", "FETCH_HEAD"]
    assert commands[2][1:4] == ["-m", "g2g", "visualize-predictions"]
    assert before == {
        path: path.read_bytes() for path in runtime.results.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("condition", ["dirty", "missing_module"])
def test_checkout_update_preserves_head_when_preflight_fails(checkout_update, condition):
    import marimo as mo

    runtime, source, git, old_commit, _, commands = checkout_update
    if condition == "dirty":
        (runtime.checkout / "README.md").write_text("local edits\n", encoding="utf-8")
        message = "thay doi local"
    else:
        source = source.replace('_repo_ref = "main"', f'_repo_ref = "{old_commit}"', 1)
        message = "chua co module"
    with pytest.raises(RuntimeError, match=message):
        exec(compile(source, "gallery-cell", "exec"), {"runtime": runtime, "mo": mo})
    assert git("-C", runtime.checkout, "rev-parse", "HEAD") == old_commit
    assert not any("visualize-predictions" in command for command in commands)
    if condition == "dirty":
        assert (runtime.checkout / "README.md").read_text() == "local edits\n"
