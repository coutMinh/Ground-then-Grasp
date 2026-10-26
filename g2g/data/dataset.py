"""Native GA++ dataset over explicit manifests; union targets never cross splits."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from g2g.data.grasp_maps import draw_grasp_maps
from g2g.data.manifest import read_manifest
from g2g.data.schema import load_grasps, load_instruction, load_mask


@lru_cache(maxsize=512)
def _cached_grasps(path: str):
    return tuple(load_grasps(path))


class GraspDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        *,
        resolution: int = 224,
        task: str = "instruction",
    ):
        if task not in {"prior", "instruction"} or resolution < 16 or resolution % 4:
            raise ValueError("Use prior/instruction mode and a resolution divisible by 4.")
        self.rows = read_manifest(manifest)
        if not self.rows or len({row["split"] for row in self.rows}) != 1:
            raise ValueError("A dataset must have one nonempty split.")
        self.root = Path(data_root)
        self.resolution, self.task = resolution, task
        self.by_image = defaultdict(list)
        for row in self.rows:
            self.by_image[row["image_id"]].append(row)
        self.items = [rows[0] for rows in self.by_image.values()] if task == "prior" else self.rows

    def __len__(self) -> int:
        return len(self.items)

    def _grasps(self, member: str):
        return _cached_grasps(str((self.root / member).resolve()))

    def __getitem__(self, index: int) -> dict:
        row = self.items[index]
        with Image.open(self.root / row["image_member"]) as source:
            raw = source.convert("RGB")
        if raw.width != raw.height:
            raise ValueError(
                "The verified adapter expects square images; do not distort rectangles."
            )
        mask = load_mask(self.root / row["mask_member"])
        if mask.shape != (raw.height, raw.width):
            raise ValueError("Mask and image dimensions differ.")
        instruction = load_instruction(self.root / row["instruction_member"])
        original_grasps = self._grasps(row["label_member"])
        union = [
            grasp
            for item in self.by_image[row["image_id"]]
            for grasp in self._grasps(item["label_member"])
        ]
        scale = self.resolution / raw.width
        targets = union if self.task == "prior" else original_grasps
        maps = draw_grasp_maps(
            [grasp.scaled(scale) for grasp in targets], (self.resolution, self.resolution)
        )
        resized = raw.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
        image = np.asarray(resized, dtype=np.float32) / 255
        image -= image.mean()  # same RGB normalization convention as LGD
        resized_mask = Image.fromarray(mask).resize(
            (self.resolution, self.resolution), Image.Resampling.NEAREST
        )
        return {
            "sample_id": row["sample_id"],
            "image_id": row["image_id"],
            "image": torch.from_numpy(image.transpose(2, 0, 1).copy()),
            "raw_image": raw,
            "text": instruction,
            "mask": torch.from_numpy(np.array(resized_mask, dtype=np.float32))[None],
            "maps": tuple(torch.from_numpy(array.copy())[None] for array in maps.arrays()),
            "gt_rects": original_grasps,
            "union_rects": union,
            "original_size": raw.width,
            "original_mask": mask,
        }


def collate_samples(samples: list[dict]) -> dict:
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "mask": torch.stack([sample["mask"] for sample in samples]),
        "maps": tuple(
            torch.stack([sample["maps"][index] for sample in samples]) for index in range(4)
        ),
        **{
            key: [sample[key] for sample in samples]
            for key in (
                "sample_id",
                "image_id",
                "raw_image",
                "text",
                "gt_rects",
                "union_rects",
                "original_size",
                "original_mask",
            )
        },
    }
