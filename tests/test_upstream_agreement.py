"""Optional upstream comparison; portable unit tests do not require this clone."""

import importlib.util
import math
import os
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

from g2g.eval.grasp_rect import GraspRectangle, match_grasp, rectangle_iou


def test_200_seeded_pairs_and_calculate_iou_match(monkeypatch):
    root = Path(
        os.environ.get(
            "LGD_REFERENCE_ROOT", Path(__file__).resolve().parents[2] / "third_party/LGD"
        )
    )
    processing = root / "utils/dataset_processing"
    if not processing.is_dir():
        pytest.skip("Set LGD_REFERENCE_ROOT to the pinned reference checkout for parity audit.")
    # Compatibility shim only inside the test: NumPy removed upstream np.float.
    monkeypatch.setattr(np, "float", float, raising=False)
    package = types.ModuleType("_g2g_lgd_reference")
    package.__path__ = [str(processing)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    modules = {}
    for name in ["grasp", "evaluation"]:
        spec = importlib.util.spec_from_file_location(
            f"{package.__name__}.{name}", processing / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        with warnings.catch_warnings():
            spec.loader.exec_module(module)
        modules[name] = module
    reference, evaluation = modules["grasp"], modules["evaluation"]
    random = np.random.default_rng(42)
    for _ in range(200):
        angle = random.uniform(-math.pi / 2, math.pi / 2)
        opening = random.uniform(12, 40)
        prediction = GraspRectangle(64, 64, opening, opening / 2, angle)
        target = GraspRectangle(
            64 + random.uniform(-20, 20),
            64 + random.uniform(-20, 20),
            random.uniform(10, 45),
            random.uniform(8, 25),
            angle + random.uniform(-1, 1),
        )
        reference_prediction = reference.Grasp((64, 64), angle, opening, opening / 2).as_gr
        reference_target = reference.Grasp(
            (target.y, target.x), target.theta, target.w, target.h
        ).as_gr
        expected_iou = reference_prediction.iou(reference_target, angle_threshold=math.pi)
        assert rectangle_iou(prediction, target) == pytest.approx(expected_iou)
        quality, angles, widths = (np.zeros((128, 128)) for _ in range(3))
        quality[64, 64], angles[64, 64], widths[64, 64] = 1, angle, opening
        expected_success = evaluation.calculate_iou_match(
            quality,
            angles,
            reference.GraspRectangles([reference_target]),
            no_grasps=1,
            grasp_width=widths,
        )
        assert match_grasp(prediction, [target]).success == expected_success
