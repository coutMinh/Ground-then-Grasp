"""Losses with explicit head semantics: prior quality is regressed, not logits."""

import torch
import torch.nn.functional as F


def prior_loss(
    predictions: tuple[torch.Tensor, ...], targets: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    if len(predictions) != 4 or len(targets) != 4:
        raise ValueError("Prior requires Q, cos(2theta), sin(2theta), width.")
    if any(pred.shape != target.shape for pred, target in zip(predictions, targets, strict=True)):
        raise ValueError("Prediction and target map shapes differ.")
    return sum(
        F.smooth_l1_loss(pred, target) for pred, target in zip(predictions, targets, strict=True)
    )


def grounder_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if logits.shape != masks.shape or logits.ndim != 4:
        raise ValueError("Grounding logits and masks must have matching Bx1xHxW shapes.")
    probabilities = logits.sigmoid()
    axes = (1, 2, 3)
    intersection = (probabilities * masks).sum(dim=axes)
    dice = (2 * intersection + 1) / (probabilities.sum(dim=axes) + masks.sum(dim=axes) + 1)
    return F.binary_cross_entropy_with_logits(logits, masks) + 1 - dice.mean()
