import math

import numpy as np
import pytest

from g2g.eval.grasp_rect import (
    GraspRectangle,
    angle_difference,
    match_grasp,
    rectangle_iou,
)


def test_corners_and_uniform_scale_roundtrip():
    grasp = GraspRectangle(23.25, 45.5, 19.75, 7.25, 0.7)
    restored = GraspRectangle.from_corners(grasp.scaled(224 / 416).corners()).scaled(416 / 224)
    np.testing.assert_allclose(restored.corners(), grasp.corners(), atol=1e-10)


def test_known_continuous_iou_and_no_overlap():
    first = GraspRectangle(10, 10, 10, 10, 0)
    second = GraspRectangle(15, 10, 10, 10, 0)
    assert rectangle_iou(first, second, "polygon") == pytest.approx(1 / 3)
    assert rectangle_iou(first, GraspRectangle(40, 40, 10, 10, 0)) == 0


def test_pixel_iou_is_translation_invariant_across_negative_coordinates():
    positive = [GraspRectangle(30, 30, 20, 10, 0), GraspRectangle(35, 30, 20, 10, 0)]
    negative = [GraspRectangle(-30, -30, 20, 10, 0), GraspRectangle(-25, -30, 20, 10, 0)]
    assert rectangle_iou(*positive) == rectangle_iou(*negative)


def test_pi_symmetry_and_angle_rejection():
    assert angle_difference(math.radians(89), math.radians(-89)) == pytest.approx(math.radians(2))
    target = GraspRectangle(50, 50, 40, 20, 0)
    assert match_grasp(GraspRectangle(50, 50, 40, 20, math.pi), [target]).success
    assert not match_grasp(GraspRectangle(50, 50, 40, 20, math.pi / 4), [target]).success


def test_iou_boundary_is_strict():
    target = GraspRectangle(10, 10, 10, 10, 0)
    assert not match_grasp(target, [target], iou_threshold=1).success
    assert match_grasp(None, [target]).success is False
    with pytest.raises(ValueError, match="positive GT"):
        match_grasp(target, [])


@pytest.mark.parametrize("values", [(0, 0, 0, 1, 0), (0, 0, 1, -1, 0), (0, 0, 1, 1, math.nan)])
def test_reject_invalid_geometry(values):
    with pytest.raises(ValueError):
        GraspRectangle(*values)
