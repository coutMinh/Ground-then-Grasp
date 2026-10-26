"""Quality/mask composition and part-aware candidate selection on 2D maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit
from skimage.feature import peak_local_max

from g2g.data.grasp_maps import GraspMaps
from g2g.eval.grasp_rect import GraspRectangle


@dataclass(frozen=True)
class ComposeConfig:
    mode: Literal["prior", "soft", "hard", "mask_peak"] = "soft"
    alpha: float = 0.5
    temperature: float = 1.0
    rerank_weight: float = 0.5
    top_k: int = 10
    min_distance: int = 5
    sigma: float = 2.0
    width_scale: float | None = None
    height_ratio: float = 0.5
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in {"prior", "soft", "hard", "mask_peak"}:
            raise ValueError("Unknown composition mode.")
        if not all(
            math.isfinite(v)
            for v in (
                self.alpha,
                self.temperature,
                self.rerank_weight,
                self.sigma,
                self.height_ratio,
                self.epsilon,
            )
        ):
            raise ValueError("Composition parameters must be finite.")
        if not 0 <= self.alpha <= 1 or self.temperature <= 0 or self.rerank_weight < 0:
            raise ValueError("Invalid alpha, temperature or re-ranking weight.")
        if self.top_k < 1 or self.min_distance < 1 or self.sigma < 0:
            raise ValueError("Invalid candidate extraction parameters.")
        if self.height_ratio <= 0 or not 0 < self.epsilon < 1:
            raise ValueError("Invalid height ratio or epsilon.")
        if self.width_scale is not None and (
            not math.isfinite(self.width_scale) or self.width_scale <= 0
        ):
            raise ValueError("Width scale must be finite and positive.")
        if self.mode == "prior" and self.rerank_weight != 0:
            raise ValueError("Prior-only mode must disable mask-based re-ranking.")


def compose_quality(
    quality: np.ndarray,
    mask: np.ndarray | None,
    config: ComposeConfig,
    *,
    mask_is_logits: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    quality = np.asarray(quality, dtype=np.float64)
    if quality.ndim != 2 or not np.isfinite(quality).all():
        raise ValueError("Quality must be a finite 2D map.")
    quality = np.clip(quality, 0, 1)  # SmoothL1 quality regression, not logits.
    if config.mode == "prior":
        return quality, None
    if mask is None or mask.shape != quality.shape or not np.isfinite(mask).all():
        raise ValueError("A finite mask with the same shape is required.")
    if mask_is_logits:
        probability = expit(mask / config.temperature)
    else:
        if np.min(mask) < 0 or np.max(mask) > 1:
            raise ValueError("Probability masks must lie in [0,1].")
        if config.temperature != 1:
            raise ValueError("Temperature applies to logits; pass mask_is_logits=True.")
        probability = mask.astype(np.float64)
    if config.mode == "hard":
        fused = quality * (probability > 0.5)
    elif config.mode == "mask_peak":
        fused = probability
    elif config.alpha == 1:
        fused = quality
    elif config.alpha == 0:
        fused = probability
    else:
        fused = np.exp(
            config.alpha * np.log(np.clip(quality, config.epsilon, 1))
            + (1 - config.alpha) * np.log(np.clip(probability, config.epsilon, 1))
        )
        fused[(quality == 0) | (probability == 0)] = 0
    return fused, probability


def detect_grasp(
    maps: GraspMaps,
    mask: np.ndarray | None = None,
    *,
    config: ComposeConfig | None = None,
    mask_is_logits: bool = False,
) -> GraspRectangle | None:
    """Return one grasp; empty hard masks/no positive quality yield no prediction.

    NMS and circular smoothing are explicit adaptations, not claimed bitwise
    reproductions of LGD post-processing. IoU evaluation is separate.
    """
    config = config or ComposeConfig()
    fused, probability = compose_quality(maps.quality, mask, config, mask_is_logits=mask_is_logits)
    smoothed = gaussian_filter(fused, config.sigma, mode="nearest")
    peaks = peak_local_max(
        smoothed,
        min_distance=config.min_distance,
        num_peaks=config.top_k,
        threshold_abs=0,
        exclude_border=False,
    )
    if not len(peaks):
        if smoothed.max() <= 0:
            return None
        # peak_local_max returns none for a constant positive map.
        peaks = np.array([np.unravel_index(np.argmax(smoothed), smoothed.shape)])
    cos = gaussian_filter(maps.cos, config.sigma, mode="nearest")
    sin = gaussian_filter(maps.sin, config.sigma, mode="nearest")
    width = gaussian_filter(maps.width, min(config.sigma, 1.0), mode="nearest")
    width_scale = config.width_scale or maps.quality.shape[1] / 2
    best_score = -math.inf
    best_grasp = None
    for row, column in peaks:
        opening = float(np.clip(width[row, column], 0, 1) * width_scale)
        if opening <= 0:
            continue
        grasp = GraspRectangle(
            float(column),
            float(row),
            opening,
            opening * config.height_ratio,
            float(np.arctan2(sin[row, column], cos[row, column]) / 2),
        )
        score = math.log(max(float(smoothed[row, column]), config.epsilon))
        if probability is not None and config.rerank_weight:
            rows, columns = grasp.inner().pixel_coordinates(probability.shape)
            mean_mask = float(probability[rows, columns].mean()) if len(rows) else 0
            score += config.rerank_weight * math.log(max(mean_mask, config.epsilon))
        if score > best_score:
            best_score, best_grasp = score, grasp
    return best_grasp
