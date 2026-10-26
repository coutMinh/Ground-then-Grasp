import pickle

import pytest

from g2g.inspection.upstream import read_split


def test_split_inventory_rejects_duplicates_and_pickle_classes(tmp_path):
    path = tmp_path / "split.obj"
    identifier = "a" * 64 + "_10"
    path.write_bytes(pickle.dumps([identifier]))
    assert read_split(path) == [identifier]
    path.write_bytes(pickle.dumps([identifier, identifier]))
    with pytest.raises(ValueError, match="Duplicate"):
        read_split(path)
    path.write_bytes(pickle.dumps(ValueError("not primitive split metadata")))
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        read_split(path)
