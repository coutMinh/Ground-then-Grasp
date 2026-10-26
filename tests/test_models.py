from types import SimpleNamespace

import pytest
import torch
from torch import nn

from g2g.losses import grounder_loss, prior_loss
from g2g.models.grconvnet3 import GraspPrior
from g2g.models.grounder import PartGrounder, parameter_counts


def test_prior_rgb_shapes_and_visual_gradients():
    model = GraspPrior(channel_size=8)
    inputs = torch.rand(2, 3, 224, 224)
    predictions = model(inputs)
    assert len(predictions) == 4
    assert all(item.shape == (2, 1, 224, 224) for item in predictions)
    prior_loss(predictions, tuple(torch.zeros_like(item) for item in predictions)).backward()
    assert model.conv1.weight.grad.abs().sum() > 0
    counts = parameter_counts(model)
    assert counts["total"] == counts["trainable"]


class StubSegmenter(nn.Module):
    """Tests the wrapper freeze contract, not pretrained CLIPSeg accuracy."""

    def __init__(self):
        super().__init__()
        self.clip = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Dropout(0.5))
        self.decoder = nn.Conv2d(4, 1, 1)

    def forward(self, pixel_values):
        return SimpleNamespace(logits=self.decoder(self.clip(pixel_values)).squeeze(1))


def test_grounder_freezes_backbone_and_trains_decoder():
    model = PartGrounder(StubSegmenter()).train()
    assert not model.model.clip.training
    logits = model({"pixel_values": torch.rand(2, 3, 32, 32)}, output_size=(24, 24))
    grounder_loss(logits, torch.ones_like(logits)).backward()
    assert all(parameter.grad is None for parameter in model.model.clip.parameters())
    assert model.model.decoder.weight.grad.abs().sum() > 0
    assert logits.shape == (2, 1, 24, 24)


def test_tiny_real_clipseg_without_downloading_weights():
    transformers = pytest.importorskip("transformers")
    config = transformers.CLIPSegConfig(
        text_config={
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 16,
            "eos_token_id": 2,
            "bos_token_id": 1,
            "pad_token_id": 0,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 8,
        },
        projection_dim=16,
        reduce_dim=16,
        extract_layers=[0, 1],
        decoder_num_attention_heads=4,
        decoder_intermediate_size=32,
    )
    model = PartGrounder(transformers.CLIPSegForImageSegmentation(config)).train()
    tokens = torch.randint(3, 100, (2, 8))
    tokens[:, -1] = 2
    logits = model(
        {
            "pixel_values": torch.rand(2, 3, 32, 32),
            "input_ids": tokens,
            "attention_mask": torch.ones_like(tokens),
        },
        output_size=(24, 24),
    )
    grounder_loss(logits, torch.ones_like(logits)).backward()
    assert all(parameter.grad is None for parameter in model.model.clip.parameters())
    assert model.model.decoder.film_mul.weight.grad.abs().sum() > 0
