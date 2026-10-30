import json
from pathlib import Path

import pytest

from g2g.data.manifest import read_manifest, write_manifest
from g2g.eval.sweep import sweep
from g2g.eval.tables import harmonic_mean, make_tables
from g2g.utils.registry import FIELDS, append_registry


def _evaluation_fixture(
    root, run_id, split, value, *, kind="benchmark_eval_prior", manifest_hash="same"
):
    directory = root / "runs" / run_id
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(
        json.dumps(
            {"status": "completed", "split": split, "system": "prior", "success_rate": value}
        )
    )
    record = dict.fromkeys(FIELDS, "synthetic-test-fixture")
    record.update(run_id=run_id, kind=kind, manifest_sha256=manifest_hash, seed=42)
    append_registry(root, record)


def test_tables_require_paired_tests_and_exclude_debug_runs(tmp_path):
    _evaluation_fixture(tmp_path, "seen", "test_seen", 0.4)
    with pytest.raises(ValueError, match="paired"):
        make_tables(str(tmp_path), ["seen"], str(tmp_path / "tables"))
    _evaluation_fixture(tmp_path, "unseen", "test_unseen", 0.2)
    report = make_tables(str(tmp_path), ["seen", "unseen"], str(tmp_path / "tables"))
    assert report["rows"][0]["harmonic"] == pytest.approx(4 / 15)
    assert harmonic_mean(0, 0) == 0
    _evaluation_fixture(tmp_path, "debug", "test_seen", 1, kind="debug_eval_prior")
    with pytest.raises(ValueError, match="never debug"):
        make_tables(str(tmp_path), ["debug"], str(tmp_path / "forbidden"))


def test_sweep_rejects_test_split_before_loading_checkpoints(tmp_path):
    source = Path(__file__).resolve().parent / "fixtures/smoke_test_seen.csv"
    rows = read_manifest(source)
    manifest = tmp_path / "test.csv"
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="val samples only"):
        sweep(str(manifest), "unused", "unused", "unused", str(tmp_path), "invalid")
