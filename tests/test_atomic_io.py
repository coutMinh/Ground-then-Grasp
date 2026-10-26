from pathlib import Path

import pytest

from g2g.utils.io import atomic_replace


def test_transient_sharing_violation_retries_without_deleting_checkpoint(tmp_path, monkeypatch):
    target, source = tmp_path / "last.pt", tmp_path / "checkpoint.tmp"
    target.write_bytes(b"previous complete checkpoint")
    source.write_bytes(b"new complete checkpoint")
    original_replace = Path.replace
    calls = []

    def temporarily_locked(path, destination):
        calls.append(path)
        assert target.read_bytes() == b"previous complete checkpoint"
        if len(calls) < 3:
            raise PermissionError("synthetic Windows sharing violation")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    monkeypatch.setattr("g2g.utils.io.time.sleep", lambda _: None)
    atomic_replace(source, target)
    assert len(calls) == 3
    assert target.read_bytes() == b"new complete checkpoint"


def test_permanent_failure_preserves_both_files(tmp_path, monkeypatch):
    target, source = tmp_path / "last.pt", tmp_path / "checkpoint.tmp"
    target.write_bytes(b"old")
    source.write_bytes(b"new")

    def locked(*args):
        raise PermissionError("synthetic permanent lock")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr("g2g.utils.io.time.sleep", lambda _: None)
    with pytest.raises(PermissionError):
        atomic_replace(source, target, attempts=3)
    assert target.read_bytes() == b"old"
    assert source.read_bytes() == b"new"
