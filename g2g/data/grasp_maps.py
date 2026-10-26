"""LGD compact-region targets with consistent width encode/decode units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g2g.eval.grasp_rect import GraspRectangle


@dataclass(frozen=True)
class GraspMaps:
    quality: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    width: np.ndarray

    def __post_init__(self) -> None:
        arrays = self.arrays()
        if arrays[0].ndim != 2 or any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("Grasp maps must be equally shaped 2D arrays.")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Grasp maps must be finite.")

    def arrays(self) -> tuple[np.ndarray, ...]:
        return self.quality, self.cos, self.sin, self.width


def draw_grasp_maps(
    grasps: list[GraspRectangle], shape: tuple[int, int], *, width_scale: float | None = None
) -> GraspMaps:
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Map shape must have two positive dimensions.")
    width_scale = width_scale if width_scale is not None else shape[1] / 2
    if not np.isfinite(width_scale) or width_scale <= 0:
        raise ValueError("Width normalization must be positive and finite.")
    quality = np.zeros(shape, dtype=np.float32)
    angle = np.zeros(shape, dtype=np.float32)
    width = np.zeros(shape, dtype=np.float32)
    for grasp in grasps:
        compact = GraspRectangle(grasp.x, grasp.y, grasp.w / 3, grasp.h, grasp.theta)
        rows, columns = compact.pixel_coordinates(shape)
        quality[rows, columns] = 1
        angle[rows, columns] = grasp.theta
        width[rows, columns] = np.clip(grasp.w / width_scale, 0, 1)
    return GraspMaps(quality, np.cos(2 * angle), np.sin(2 * angle), width)
