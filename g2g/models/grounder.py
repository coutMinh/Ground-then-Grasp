"""CLIPSeg decoder adaptation with a frozen, evaluation-mode CLIP backbone."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class PartGrounder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.clip.requires_grad_(False)
        self.model.clip.eval()

    @classmethod
    def from_pretrained(
        cls, model_id: str = "CIDAS/clipseg-rd64-refined", *, revision: str = "main"
    ) -> PartGrounder:
        from transformers import CLIPSegForImageSegmentation

        model = CLIPSegForImageSegmentation.from_pretrained(model_id, revision=revision)
        return cls(model)

    def train(self, mode: bool = True) -> PartGrounder:
        super().train(mode)
        self.model.clip.eval()
        return self

    def forward(
        self, inputs: dict[str, Any], *, output_size: tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        logits = self.model(**inputs).logits
        return F.interpolate(
            logits[:, None], size=output_size, mode="bilinear", align_corners=False
        )


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
