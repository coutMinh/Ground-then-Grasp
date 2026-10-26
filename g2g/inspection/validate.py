"""Validate every native sample before handing a manifest to a trainer."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from g2g.data.manifest import read_manifest
from g2g.data.schema import load_grasps, load_instruction, load_mask
from g2g.utils.io import sha256_file, write_json


def validate_data(manifest: str, data_root: str, output: str) -> dict:
    rows = read_manifest(manifest)
    root = Path(data_root)
    failures = []
    images = set()
    for row in rows:
        try:
            with Image.open(root / row["image_member"]) as image:
                image.load()
                size = image.size
            if size[0] != size[1]:
                raise ValueError("Non-square image is outside verified schema.")
            mask = load_mask(root / row["mask_member"])
            if mask.shape != (size[1], size[0]):
                raise ValueError("Image/mask size mismatch.")
            load_grasps(root / row["label_member"])
            load_instruction(root / row["instruction_member"])
            images.add(row["image_id"])
        except (ValueError, OSError, RuntimeError) as error:
            failures.append({"sample_id": row["sample_id"], "error_type": type(error).__name__})
    report = {
        "manifest_sha256": sha256_file(manifest),
        "rows": len(rows),
        "decoded_images": len(images),
        "failure_count": len(failures),
        "failures": failures,
    }
    write_json(output, report)
    if failures:
        raise ValueError(
            f"{len(failures)} invalid samples; see the validation report. No rows removed."
        )
    return report
