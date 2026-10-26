"""Small explicit I/O helpers shared by command-line entry points."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml


def atomic_replace(source: str | Path, target: str | Path, *, attempts: int = 8) -> None:
    """Publish a complete file, retrying transient Windows sharing violations.

    A failed replacement leaves the previous target intact and retains the
    temporary file. Never delete a checkpoint just to make replacement succeed.
    """
    if attempts < 1:
        raise ValueError("Replacement attempts must be positive.")
    source, target = Path(source), Path(target)
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.02 * 2**attempt, 0.2))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    atomic_replace(temporary, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config
