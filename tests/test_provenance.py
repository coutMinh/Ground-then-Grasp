import csv
import random

import numpy as np
import pytest
import torch

from g2g.utils.registry import FIELDS, append_registry
from g2g.utils.seed import capture_rng, restore_rng, seed_everything


def test_rng_checkpoint_is_weights_only_compatible(tmp_path):
    seed_everything(42)
    path = tmp_path / "state.pt"
    torch.save(capture_rng(), path)
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(torch.load(path, weights_only=True))
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


def test_registry_duplicate_does_not_leave_stale_lock(tmp_path):
    record = dict.fromkeys(FIELDS, "fixture")
    append_registry(tmp_path, record)
    with pytest.raises(ValueError, match="already registered"):
        append_registry(tmp_path, record)
    assert not (tmp_path / "registry.lock").exists()
    record["run_id"] = "next-fixture"
    append_registry(tmp_path, record)
    with (tmp_path / "registry.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
