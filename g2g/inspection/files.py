"""Inspect supplied local dataset files without assuming a GA++ native schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from g2g.inspection.upstream import PrimitiveUnpickler
from g2g.utils.io import sha256_file, write_json


def describe(value: object, depth: int = 0) -> dict:
    result = {"type": type(value).__name__}
    if isinstance(value, np.ndarray):
        result.update(shape=list(value.shape), dtype=str(value.dtype))
        if value.size and np.issubdtype(value.dtype, np.number):
            finite = value[np.isfinite(value)]
            result["finite_count"] = int(finite.size)
            if finite.size:
                result.update(min=float(finite.min()), max=float(finite.max()))
        result["preview"] = repr(value.reshape(-1)[:12].tolist())
    elif isinstance(value, (list, tuple, dict)):
        result["length"] = len(value)
        if depth < 3:
            if isinstance(value, dict):
                result["items"] = [
                    {"key": str(key), "value": describe(item, depth + 1)}
                    for key, item in list(value.items())[:5]
                ]
            else:
                result["items"] = [describe(item, depth + 1) for item in value[:5]]
    elif isinstance(value, (str, int, float, bool)) or value is None:
        result["preview"] = repr(value)[:500]
    return result


def inspect_files(paths: list[str], output: str) -> dict:
    report = {"scope": "Local file inspection only; no inferred filename mapping.", "files": []}
    for raw_path in paths:
        path = Path(raw_path)
        entry = {"path": path.as_posix(), "sha256": sha256_file(path)}
        if path.suffix.lower() == ".npy":
            entry["content"] = describe(np.load(path, allow_pickle=False))
        elif path.suffix.lower() == ".pkl":
            with path.open("rb") as stream:
                entry["content"] = describe(PrimitiveUnpickler(stream).load())
        elif path.suffix.lower() == ".pt":
            import torch

            value = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(value, torch.Tensor):
                value = value.detach().numpy()
            entry["content"] = describe(value)
        elif path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            with Image.open(path) as image:
                entry["content"] = {"size": list(image.size), "mode": image.mode}
        else:
            raise ValueError(f"Unsupported inspection extension: {path.suffix}")
        report["files"].append(entry)
    write_json(output, report)
    return report
