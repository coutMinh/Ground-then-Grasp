import numpy as np
import pytest

from g2g.compose import ComposeConfig, compose_quality, detect_grasp
from g2g.data.grasp_maps import GraspMaps, draw_grasp_maps
from g2g.eval.grasp_rect import GraspRectangle, rectangle_iou


def test_quality_regression_does_not_turn_zero_background_into_half():
    quality = np.array([[0, 1]], dtype=float)
    fused, _ = compose_quality(quality, np.ones_like(quality), ComposeConfig())
    np.testing.assert_array_equal(fused, [[0, 1]])


def test_alpha_endpoints_and_hard_mask():
    quality = np.array([[0.2, 0.8]])
    mask = np.array([[0.9, 0.1]])
    for alpha, expected in [(0, mask), (1, quality)]:
        result, _ = compose_quality(quality, mask, ComposeConfig(alpha=alpha))
        np.testing.assert_array_equal(result, expected)
    result, _ = compose_quality(quality, mask, ComposeConfig(mode="hard"))
    np.testing.assert_array_equal(result, [[0.2, 0]])


def test_extreme_logits_are_finite_and_temperature_requires_logits():
    quality = np.ones((1, 2))
    result, _ = compose_quality(
        quality, np.array([[-10000.0, 10000.0]]), ComposeConfig(), mask_is_logits=True
    )
    assert np.isfinite(result).all()
    with pytest.raises(ValueError, match="Temperature"):
        compose_quality(quality, quality, ComposeConfig(temperature=2))


def test_width_units_and_compatible_decoder_roundtrip():
    target = GraspRectangle(48, 48, 30, 15, 0)
    maps = draw_grasp_maps([target], (96, 96))
    assert maps.width[48, 48] == pytest.approx(30 / 48)
    prediction = detect_grasp(maps, config=ComposeConfig(mode="prior", rerank_weight=0))
    assert prediction is not None
    assert rectangle_iou(prediction, target, "polygon") > 0.9


def test_compact_target_preserves_full_height_and_clips_opening():
    maps = draw_grasp_maps([GraspRectangle(32, 32, 48, 12, 0)], (64, 64))
    assert maps.quality[32, 39] == 1
    assert maps.quality[32, 42] == 0
    assert maps.quality[37, 32] == 1
    assert maps.width[32, 32] == 1
    assert maps.cos[0, 0] == 1


def test_empty_hard_mask_returns_no_grasp():
    maps = draw_grasp_maps([GraspRectangle(32, 32, 20, 10, 0)], (64, 64))
    assert detect_grasp(maps, np.zeros((64, 64)), config=ComposeConfig(mode="hard")) is None


def test_mask_selects_the_referred_candidate():
    quality = np.zeros((64, 64), dtype=np.float32)
    quality[20, 20] = 0.9
    quality[44, 44] = 0.8
    mask = np.zeros_like(quality)
    mask[40:49, 40:49] = 1
    maps = GraspMaps(
        quality, np.ones_like(quality), np.zeros_like(quality), np.full_like(quality, 0.4)
    )
    result = detect_grasp(maps, mask, config=ComposeConfig(sigma=0, rerank_weight=0))
    assert result is not None
    assert (result.x, result.y) == (44, 44)


def test_reranking_can_prefer_candidate_with_more_part_support():
    quality = np.zeros((64, 64))
    quality[20, 20], quality[44, 44] = 0.9, 0.8
    mask = np.zeros_like(quality)
    mask[20, 20] = 1
    mask[38:51, 38:51] = 1
    maps = GraspMaps(
        quality, np.ones_like(quality), np.zeros_like(quality), np.full_like(quality, 0.5)
    )
    without = detect_grasp(maps, mask, config=ComposeConfig(sigma=0, rerank_weight=0))
    with_rerank = detect_grasp(maps, mask, config=ComposeConfig(sigma=0, rerank_weight=1))
    assert (without.x, without.y) == (20, 20)
    assert (with_rerank.x, with_rerank.y) == (44, 44)
