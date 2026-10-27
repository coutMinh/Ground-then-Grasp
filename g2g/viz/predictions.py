"""Render saved instruction-level predictions against their native ground truth.

No model inference is performed. Samples come from the evaluation's manifest
snapshot; mixed sampling intentionally includes successes and failures.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import textwrap
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from g2g.data.manifest import read_manifest
from g2g.data.schema import load_grasps, load_instruction
from g2g.eval.grasp_rect import GraspRectangle, match_grasp
from g2g.utils.io import sha256_file, write_json

GREEN = (0, 190, 95)
RED = (235, 55, 55)


def select_samples(records: list[dict], limit: int, selection: str, seed: int) -> list[dict]:
    if limit < 1 or selection not in {"mixed", "random", "success", "failure"}:
        raise ValueError("Use limit >= 1 and selection mixed/random/success/failure.")
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda row: row["sample_id"])
    rng.shuffle(ordered)
    if selection in {"success", "failure"}:
        matching = [row for row in ordered if bool(row["success"]) == (selection == "success")]
        return matching[:limit]
    if selection == "random":
        return ordered[:limit]
    successes = [row for row in ordered if row["success"]]
    failures = [row for row in ordered if not row["success"]]
    selected = successes[: (limit + 1) // 2] + failures[: limit // 2]
    selected_ids = {row["sample_id"] for row in selected}
    selected += [row for row in ordered if row["sample_id"] not in selected_ids][
        : max(0, limit - len(selected))
    ]
    return selected


def _font():
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, 16)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_grasp(panel: Image.Image, grasp: GraspRectangle, color, scale: float):
    points = [tuple(point) for point in grasp.corners() * scale]
    draw = ImageDraw.Draw(panel)
    draw.line([*points, points[0]], fill=color, width=3)
    center = (grasp.x * scale, grasp.y * scale)
    draw.ellipse((center[0] - 2, center[1] - 2, center[0] + 2, center[1] + 2), fill=color)


def _render_row(image_path: Path, instruction: str, targets: list, prediction, record: dict):
    with Image.open(image_path) as source:
        raw = source.convert("RGB")
    panel_width = max(320, raw.width)
    scale = panel_width / raw.width
    panel_height = round(raw.height * scale)
    raw = raw.resize((panel_width, panel_height), Image.Resampling.BILINEAR)
    gt_panel, prediction_panel, combined_panel = raw.copy(), raw.copy(), raw.copy()
    for target in targets:
        _draw_grasp(gt_panel, target, GREEN, scale)
        _draw_grasp(combined_panel, target, GREEN, scale)
    if prediction is not None:
        _draw_grasp(prediction_panel, prediction, RED, scale)
        _draw_grasp(combined_panel, prediction, RED, scale)
    instruction_lines = textwrap.wrap(instruction, width=max(40, panel_width * 3 // 9))
    header_height = 32 + 24 * len(instruction_lines) + 32
    canvas = Image.new("RGB", (panel_width * 3, header_height + panel_height + 60), "white")
    draw, font = ImageDraw.Draw(canvas), _font()
    label = "SUCCESS" if record["success"] else "FAILURE"
    angle = record.get("angle_difference")
    angle_text = f"{math.degrees(angle):.1f} deg" if angle is not None else "no eligible angle"
    if prediction is None:
        angle_text = "n/a (no prediction)"
    draw.text((12, 7), f"{label} | sample_id: {record['sample_id']}", fill="black", font=font)
    for number, line in enumerate(instruction_lines):
        draw.text((12, 32 + 24 * number), line, fill="black", font=font)
    title_y = header_height - 26
    titles = ("Ground truth (green)", "Prediction (red)", "GT + prediction")
    for index, (panel, title) in enumerate(
        zip((gt_panel, prediction_panel, combined_panel), titles, strict=True)
    ):
        draw.text((index * panel_width + 12, title_y), title, fill="black", font=font)
        canvas.paste(panel, (index * panel_width, header_height))
    footer = header_height + panel_height + 8
    draw.text(
        (12, footer),
        f"Saved angle-eligible grasp IoU: {record['best_iou']:.4f} | Angle: {angle_text}",
        fill="black",
        font=font,
    )
    if prediction is not None:
        details = (
            f"Prediction (native pixels): x={prediction.x:.1f}, y={prediction.y:.1f}, "
            f"w={prediction.w:.1f}, h={prediction.h:.1f}, "
            f"theta={math.degrees(prediction.theta):.1f} deg"
        )
    else:
        details = "No grasp predicted."
    draw.text((12, footer + 24), details, fill="black", font=font)
    return canvas


def visualize_predictions(
    run_dir: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    limit: int = 6,
    seed: int = 42,
    selection: str = "mixed",
) -> dict:
    run_dir, data_root, output = Path(run_dir), Path(data_root), Path(output)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics["status"] != "completed":
        raise ValueError("Select a completed evaluation run.")
    if config.get("target_scope", "instruction") != "instruction":
        raise ValueError("This viewer expects instruction-level evaluation, not scene union.")
    manifest = read_manifest(run_dir / "manifest.csv")
    by_id = {row["sample_id"]: row for row in manifest}
    records = [
        json.loads(line)
        for line in (run_dir / "preds.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("The evaluation contains no saved predictions.")
    if len(records) != len({row["sample_id"] for row in records}):
        raise ValueError("Duplicate sample IDs in predictions.")
    if {row["sample_id"] for row in records} != set(by_id):
        raise ValueError(
            "Predictions and the saved evaluation manifest refer to different samples."
        )
    if len(records) != metrics["count"] or not math.isclose(
        sum(row["success"] for row in records) / len(records),
        metrics["success_rate"],
        abs_tol=1e-12,
    ):
        raise ValueError("Saved predictions disagree with evaluation count/success_rate.")
    chosen = select_samples(records, limit, selection, seed)
    if not chosen:
        raise ValueError("No predictions match the requested selection.")
    output.mkdir(parents=True, exist_ok=False)
    panels, samples = [], []
    for index, record in enumerate(chosen):
        row = by_id[record["sample_id"]]
        targets = load_grasps(data_root / row["label_member"])
        instruction = load_instruction(data_root / row["instruction_member"])
        if record.get("instruction") != instruction:
            raise ValueError("Saved instruction disagrees with the native sample.")
        prediction = GraspRectangle(**record["rectangle"]) if record["rectangle"] else None
        checked = match_grasp(
            prediction,
            targets,
            iou_threshold=config.get("iou_threshold", 0.25),
            iou_mode=config.get("iou_mode", "pixel"),
        )
        panel = _render_row(
            data_root / row["image_member"], instruction, targets, prediction, record
        )
        filename = output / f"{index + 1:02d}-{row['sample_id']}.png"
        panel.save(filename)
        panels.append(panel)
        samples.append(
            {
                "sample_id": row["sample_id"],
                "instruction": instruction,
                "success": record["success"],
                "grasp_iou": record["best_iou"],
                "angle_deg": (
                    math.degrees(record["angle_difference"])
                    if record["angle_difference"] is not None
                    else None
                ),
                "prediction": record["rectangle"],
                "gt_count": len(targets),
                "recomputed_success": checked.success,
                "recomputed_iou": checked.best_iou,
                "matches_saved_metric": checked.success == record["success"]
                and math.isclose(checked.best_iou, record["best_iou"], abs_tol=1e-9),
                "figure": str(filename.resolve()),
            }
        )
    sheet_size = (max(panel.width for panel in panels), sum(panel.height for panel in panels))
    sheet = Image.new("RGB", sheet_size, "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top))
        top += panel.height
    sheet_path = output / "contact_sheet.png"
    sheet.save(sheet_path)
    population = Counter("success" if record["success"] else "failure" for record in records)
    report = {
        "run_id": run_dir.name,
        "system": config["system"],
        "split": metrics["split"],
        "selection": selection,
        "seed": seed,
        "count": len(samples),
        "selection_note": (
            "Seeded successes and failures for illustration; "
            "not a representative accuracy estimate."
            if selection == "mixed"
            else "Seeded selection; full evaluation metrics remain unchanged."
        ),
        "population": dict(population),
        "predictions_sha256": sha256_file(run_dir / "preds.jsonl"),
        "manifest_sha256": sha256_file(run_dir / "manifest.csv"),
        "evaluation_config_sha256": sha256_file(run_dir / "config.json"),
        "contact_sheet": str(sheet_path.resolve()),
        "samples": samples,
    }
    write_json(output / "selection.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection", choices=["mixed", "random", "success", "failure"], default="mixed"
    )
    args = parser.parse_args()
    result = visualize_predictions(
        args.run_dir,
        args.data_root,
        args.output,
        limit=args.limit,
        seed=args.seed,
        selection=args.selection,
    )
    summary = {key: result[key] for key in ("run_id", "count", "contact_sheet")}
    print(json.dumps(summary, indent=2))
