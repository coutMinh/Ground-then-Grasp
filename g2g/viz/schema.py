"""Render matched real-file overlays for the schema verification gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from g2g.data.schema import SampleKey, load_grasps, load_instruction, load_mask
from g2g.utils.io import write_json


def schema_overlays(data_root: str, output: str, limit: int = 10) -> dict:
    root, destination = Path(data_root), Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    samples = selection["sample_ids"]
    tiles = []
    records = []
    for index, sample_id in enumerate(samples):
        key = SampleKey.parse(sample_id)
        members = key.members()
        with Image.open(root / members["image"]) as source:
            image = source.convert("RGB")
        mask = load_mask(root / members["mask"])
        grasps = load_grasps(root / members["label"])
        instruction = load_instruction(root / members["instruction"])
        if mask.shape != (image.height, image.width):
            raise ValueError(f"Image/mask dimensions do not match: {sample_id}")
        centers_in_mask = sum(
            0 <= round(g.y) < image.height
            and 0 <= round(g.x) < image.width
            and mask[round(g.y), round(g.x)]
            for g in grasps
        )
        records.append(
            {
                "sample_id": sample_id,
                "image_size": list(image.size),
                "grasp_count": len(grasps),
                "mask_pixels": int(mask.sum()),
                "centers_in_mask": int(centers_in_mask),
                "instruction": instruction,
            }
        )
        if index >= limit:
            continue
        pixels = np.asarray(image).copy()
        pixels[mask] = (0.6 * pixels[mask] + 0.4 * np.array([0, 220, 90])).astype(np.uint8)
        overlay = Image.new("RGB", (image.width, image.height + 46), "white")
        overlay.paste(Image.fromarray(pixels), (0, 46))
        draw = ImageDraw.Draw(overlay)
        draw.text((8, 6), instruction, fill="black")
        draw.text((8, 23), f"{sample_id[:12]}  positive grasps={len(grasps)}", fill="black")
        for grasp in grasps:
            points = [(float(x), float(y + 46)) for x, y in grasp.corners()]
            draw.line([*points, points[0]], fill=(255, 220, 0), width=2)
            draw.ellipse((grasp.x - 2, grasp.y + 44, grasp.x + 2, grasp.y + 48), fill="red")
        overlay.save(destination / f"schema_check_{index:02d}.png")
        tiles.append(overlay)
    if tiles:
        columns = 2
        contact = Image.new(
            "RGB",
            (tiles[0].width * columns, tiles[0].height * ((len(tiles) + columns - 1) // columns)),
            "white",
        )
        for index, tile in enumerate(tiles):
            contact.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
        contact.save(destination / "contact_sheet.jpg", quality=88)
    report = {
        "parsed_samples": len(records),
        "rendered_overlays": len(tiles),
        "visually_reviewed": False,
        "samples": records,
    }
    write_json(destination / "verification.json", report)
    return report
