"""Run provenance, including failed/debug runs, with process-safe CSV writes."""

from __future__ import annotations

import csv
import platform
import re
import subprocess
from pathlib import Path

from filelock import FileLock

FIELDS = [
    "run_id",
    "kind",
    "git_commit",
    "git_dirty",
    "config_path",
    "manifest_sha256",
    "seed",
    "hardware",
    "wall_seconds",
    "metrics_path",
]


def git_provenance(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[2]
    command = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    try:
        commit = subprocess.check_output(
            [*command, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output([*command, "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "uncommitted", True
    return {"git_commit": commit, "git_dirty": dirty}


def hardware_name() -> str:
    import torch

    return (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.machine() + " CPU"
    )


def run_directory(root: str | Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id):
        raise ValueError("Invalid run_id; use a short filename without paths.")
    directory = Path(root) / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def append_registry(root: str | Path, record: dict) -> None:
    if set(record) != set(FIELDS):
        raise ValueError("Registry record has missing or unexpected fields.")
    path = Path(root) / "registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prevent two writers from corrupting an experiment registry.
    # OS-backed locking avoids Windows races around creating/deleting lock files.
    # The sidecar may remain on disk after release; it is not itself a held lock.
    with FileLock(str(path.with_suffix(".lock")), timeout=5):
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != FIELDS:
                    raise ValueError("Existing registry has an incompatible schema.")
                if any(row["run_id"] == record["run_id"] for row in reader):
                    raise ValueError("Run already registered; choose a new run_id.")
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            if new_file:
                writer.writeheader()
            writer.writerow(record)
