"""Native GA++ adapter, based on inspected files and pinned LGD geometry.

See docs/REAL_DATA_VERIFICATION.md. Parsing is strict: unsupported schema
changes fail visibly instead of silently switching conventions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from g2g.eval.grasp_rect import GraspRectangle
from g2g.inspection.upstream import PrimitiveUnpickler

SAMPLE_ID = re.compile(r"^(?P<image>[0-9a-f]{64})_(?P<object>\d+)_(?P<part>\d+)$")


@dataclass(frozen=True)
class SampleKey:
    image_id: str
    object_index: int
    part_index: int

    @classmethod
    def parse(cls, sample_id: str) -> SampleKey:
        match = SAMPLE_ID.fullmatch(sample_id)
        if match is None:
            raise ValueError("Expected <64-hex-image-id>_<object-index>_<part-index>.")
        return cls(match["image"], int(match["object"]), int(match["part"]))

    @property
    def sample_id(self) -> str:
        return f"{self.image_id}_{self.object_index}_{self.part_index}"

    @property
    def object_id(self) -> str:
        return f"{self.image_id}_{self.object_index}"

    def members(self) -> dict[str, str]:
        return {
            "image": f"image/{self.image_id}.jpg",
            "label": f"grasp_label_positive/{self.sample_id}.pt",
            "mask": f"part_mask/{self.sample_id}.npy",
            "instruction": f"grasp_instructions/{self.sample_id}.pkl",
        }


def load_grasps(path: str | Path) -> list[GraspRectangle]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 6:
        raise ValueError("Expected a native Nx6 grasp tensor.")
    if not value.is_floating_point() or not torch.isfinite(value).all() or len(value) == 0:
        raise ValueError("Positive labels must be a nonempty finite floating tensor.")
    return [
        GraspRectangle(x, y, w, h, -math.radians(theta)) for _, x, y, w, h, theta in value.tolist()
    ]


def load_instruction(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        value = PrimitiveUnpickler(stream).load()
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty instruction string.")
    return value.strip()


def load_mask(path: str | Path) -> np.ndarray:
    mask = np.load(path, allow_pickle=False)
    if mask.ndim != 2 or mask.dtype not in (np.dtype("uint8"), np.dtype("bool")):
        raise ValueError("Expected a 2D uint8/bool part mask.")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected binary part-mask values 0/1.")
    return mask.astype(bool)
