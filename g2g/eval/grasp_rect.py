"""Rectangle geometry; angles are radians, CCW visually, coordinates are pixels.

Corner ordering follows LGD's Grasp.as_gr, expressed as (x,y) instead of (y,x).
See docs/DATA_SCHEMA.md and THIRD_PARTY.md for the upstream convention.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon


def angle_difference(left: float, right: float) -> float:
    """Absolute difference modulo pi, in [0,pi/2]."""
    return abs((left - right + math.pi / 2) % math.pi - math.pi / 2)


@dataclass(frozen=True)
class GraspRectangle:
    x: float
    y: float
    w: float
    h: float
    theta: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Rectangle values must be finite.")
        if self.w <= 0 or self.h <= 0:
            raise ValueError("Rectangle dimensions must be positive.")

    def corners(self) -> np.ndarray:
        along = np.array([math.cos(self.theta), -math.sin(self.theta)]) * self.w / 2
        across = np.array([math.sin(self.theta), math.cos(self.theta)]) * self.h / 2
        center = np.array([self.x, self.y])
        return np.stack(
            [
                center - along - across,
                center + along - across,
                center + along + across,
                center - along + across,
            ]
        )

    @classmethod
    def from_corners(cls, points: np.ndarray) -> GraspRectangle:
        points = np.asarray(points, dtype=np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Expected four finite (x,y) corners in rectangle order.")
        along = points[1] - points[0]
        across = points[2] - points[1]
        if (
            not np.allclose(points[2] - points[3], along, atol=1e-6)
            or not np.allclose(points[3] - points[0], across, atol=1e-6)
            or not np.isclose(np.dot(along, across), 0, atol=1e-6)
        ):
            raise ValueError("Corners must form an ordered rectangle.")
        center = points.mean(axis=0)
        return cls(
            float(center[0]),
            float(center[1]),
            float(np.linalg.norm(along)),
            float(np.linalg.norm(across)),
            math.atan2(-along[1], along[0]),
        )

    def scaled(self, factor: float) -> GraspRectangle:
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Scale must be finite and positive.")
        return GraspRectangle(
            self.x * factor, self.y * factor, self.w * factor, self.h * factor, self.theta
        )

    def inner(self, factor: float = 0.5) -> GraspRectangle:
        if not 0 < factor <= 1:
            raise ValueError("Inner rectangle scale must lie in (0,1].")
        return GraspRectangle(self.x, self.y, self.w * factor, self.h * factor, self.theta)

    def pixel_coordinates(
        self, shape: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        points = self.corners()
        if shape is not None:
            return polygon(points[:, 1], points[:, 0], shape=shape)
        # skimage clips negative coordinates even without shape. Translate by
        # whole pixels first so unbounded IoU remains translation invariant.
        offset = np.minimum(np.floor(points.min(axis=0)), 0).astype(int)
        shifted = points - offset
        rows, columns = polygon(shifted[:, 1], shifted[:, 0])
        return rows + offset[1], columns + offset[0]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def rectangle_iou(
    left: GraspRectangle, right: GraspRectangle, mode: Literal["pixel", "polygon"] = "pixel"
) -> float:
    if mode == "polygon":
        first, second = Polygon(left.corners()), Polygon(right.corners())
        return float(first.intersection(second).area / first.union(second).area)
    if mode != "pixel":
        raise ValueError(f"Unknown IoU mode: {mode}")
    left_rows, left_cols = left.pixel_coordinates()
    right_rows, right_cols = right.pixel_coordinates()
    # Coordinate sets reproduce raster IoU without a canvas allocation or negative-index wrap.
    first = set(zip(left_rows.tolist(), left_cols.tolist(), strict=True))
    second = set(zip(right_rows.tolist(), right_cols.tolist(), strict=True))
    union = first | second
    return len(first & second) / len(union) if union else 0.0


@dataclass(frozen=True)
class GraspMatch:
    success: bool
    best_iou: float
    angle_difference: float | None
    gt_index: int | None


def match_grasp(
    prediction: GraspRectangle | None,
    ground_truth: list[GraspRectangle],
    *,
    iou_threshold: float = 0.25,
    angle_threshold: float = math.pi / 6,
    iou_mode: Literal["pixel", "polygon"] = "pixel",
) -> GraspMatch:
    """LGD boundaries: IoU strictly > threshold, angle difference <= threshold.

    best_iou is the best *angle-eligible* IoU used by the success decision.
    Empty GT is invalid data, while a missing prediction is a recorded failure.
    """
    if not ground_truth:
        raise ValueError("A benchmark sample must contain at least one positive GT grasp.")
    if not 0 <= iou_threshold <= 1 or not 0 <= angle_threshold <= math.pi / 2:
        raise ValueError("Invalid metric thresholds.")
    if prediction is None:
        return GraspMatch(False, 0.0, None, None)
    best = GraspMatch(False, 0.0, None, None)
    for index, target in enumerate(ground_truth):
        delta = angle_difference(prediction.theta, target.theta)
        if delta > angle_threshold:
            continue
        score = rectangle_iou(prediction, target, mode=iou_mode)
        if best.gt_index is None or score > best.best_iou:
            best = GraspMatch(score > iou_threshold, score, delta, index)
    return best
