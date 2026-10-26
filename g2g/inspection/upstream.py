"""Audit official split files without allowing pickle code execution."""

from __future__ import annotations

import pickle
import re
import subprocess
from pathlib import Path

from g2g.utils.io import sha256_file, write_json

LGD_COMMIT = "501904dc5c30c8fad58d90f7e4f49b4095f11eb0"
OBJECT_ID = re.compile(r"^(?P<image>[0-9a-f]{64})_(?P<object>\d+)$")


class PrimitiveUnpickler(pickle.Unpickler):
    """Only Python primitives are accepted; no imports or persistent objects."""

    def find_class(self, module: str, name: str) -> None:
        raise pickle.UnpicklingError(f"Pickle class is not allowed: {module}.{name}")

    def persistent_load(self, pid: object) -> None:
        raise pickle.UnpicklingError("Persistent pickle references are not allowed.")


def read_split(path: str | Path) -> list[str]:
    with Path(path).open("rb") as stream:
        entries = PrimitiveUnpickler(stream).load()
    if not isinstance(entries, list) or not all(
        isinstance(entry, str) and OBJECT_ID.fullmatch(entry) for entry in entries
    ):
        raise ValueError(f"Expected list of <64-hex-image-id>_<object-id>: {path}")
    if len(entries) != len(set(entries)):
        raise ValueError(f"Duplicate object IDs in split: {path}")
    return entries


def audit_upstream(root: str | Path, output: str | Path) -> dict:
    root = Path(root).resolve()
    # A command-local exception handles Windows sandbox ownership without changing Git config.
    command = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    commit = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    if commit != LGD_COMMIT:
        raise ValueError(f"Expected LGD commit {LGD_COMMIT}, received {commit}.")
    groups = {}
    report = {"repository": "https://github.com/Fsoft-AIC/LGD", "commit": commit, "splits": {}}
    for path in sorted((root / "split/grasp-anything++").rglob("*.obj")):
        entries = read_split(path)
        name = path.relative_to(root).as_posix()
        images = {entry.rsplit("_", 1)[0] for entry in entries}
        groups[name] = (set(entries), images)
        report["splits"][name] = {
            "sha256": sha256_file(path),
            "entries": len(entries),
            "unique_images": len(images),
            "id_level": "object",
            "examples": entries[:3],
        }
    report["overlap"] = []
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            report["overlap"].append(
                {
                    "left": left,
                    "right": right,
                    "object_ids": len(groups[left][0] & groups[right][0]),
                    "image_ids": len(groups[left][1] & groups[right][1]),
                }
            )
    report["scope"] = "Official split metadata, not verified archive row counts."
    write_json(output, report)
    return report
