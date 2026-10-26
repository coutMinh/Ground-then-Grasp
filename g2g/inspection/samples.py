"""Fetch a small matched sample set for VERIFY, before writing a native adapter."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from g2g.data.remote_zip import HTTPRangeReader
from g2g.inspection.files import inspect_files
from g2g.inspection.upstream import read_split
from g2g.utils.io import sha256_file, write_json

GA_REVISION = "6b4917455d0910c23f936a6c7a968418ba518897"
GAPP_REVISION = "9b30c841d4f04e4a62e9f7abb3b503337a6200a7"


def archive_urls(kind: str) -> list[str]:
    if kind == "images":
        base = f"https://huggingface.co/datasets/airvlab/Grasp-Anything/resolve/{GA_REVISION}"
        return [f"{base}/image_part_aa", f"{base}/image_part_ab"]
    base = f"https://huggingface.co/datasets/airvlab/Grasp-Anything-pp/resolve/{GAPP_REVISION}"
    return [f"{base}/{kind}.zip"]


def _fetch_stage(
    kind: str, root: Path, sample_ids: list[str], image_ids: list[str], count: int | None = None
) -> dict:
    suffix = {
        "grasp_label_positive": ".pt",
        "grasp_instructions": ".pkl",
        "part_mask": ".npy",
        "images": ".jpg",
    }[kind]
    prefix = "image" if kind == "images" else kind
    ids = image_ids if kind == "images" else sample_ids
    extracted = []
    missing = []
    print(f"Reading {kind} archive index...", flush=True)
    with (
        HTTPRangeReader(
            archive_urls(kind),
            block_size=64 * 1024,
            cache_dir=root.parent / "inspection/http-cache",
            token=os.environ.get("HF_TOKEN"),
        ) as reader,
        zipfile.ZipFile(reader) as archive,
    ):
        total = len(archive.infolist())
        for identifier in ids:
            member = f"{prefix}/{identifier}{suffix}"
            if member not in archive.NameToInfo and kind == "grasp_label_positive":
                missing.append(member)
                continue
            content = archive.read(member)  # zipfile checks each member CRC.
            target = root / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            extracted.append(
                {"member": member, "sha256": sha256_file(target), "bytes": len(content)}
            )
            if count is not None and len(extracted) == count:
                break
        if count is not None and len(extracted) != count:
            raise ValueError(
                "Not enough existing part-0 members for the requested VERIFY sample set."
            )
    print(f"Fetched {len(extracted)} {kind} members (CRC checked).", flush=True)
    return {"archive_entries": total, "extracted": extracted, "skipped_missing": missing}


def fetch_verification_samples(upstream: str, destination: str, count: int = 20) -> dict:
    if count < 1:
        raise ValueError("Sample count must be positive.")
    root = Path(destination).resolve()
    split_root = Path(upstream) / "split/grasp-anything++"
    train = read_split(split_root / "train/seen.obj")
    test_images = {
        identifier.rsplit("_", 1)[0]
        for filename in ("seen.obj", "unseen.obj")
        for identifier in read_split(split_root / "test" / filename)
    }
    eligible = [
        identifier for identifier in train if identifier.rsplit("_", 1)[0] not in test_images
    ]
    # Select one candidate part 0 per distinct train image. Existence is verified
    # against archive.getinfo/read; missing members fail rather than being guessed.
    selected = []
    images = []
    for object_id in eligible:
        image_id = object_id.rsplit("_", 1)[0]
        if image_id not in images:
            images.append(image_id)
            selected.append(f"{object_id}_0")
        if len(selected) == count * 3:
            break
    report = {
        "scope": "VERIFY sample set; not a train/val/test benchmark manifest",
        "ga_revision": GA_REVISION,
        "gapp_revision": GAPP_REVISION,
        "selection_rule": (
            "first eligible official-train object per image with existing part-0 label; VERIFY only"
        ),
        "sample_ids": selected,
        "stages": {},
    }
    for kind in ("grasp_label_positive", "grasp_instructions", "part_mask", "images"):
        report["stages"][kind] = _fetch_stage(
            kind, root, selected, images, count=count if kind == "grasp_label_positive" else None
        )
        if kind == "grasp_label_positive":
            selected = [Path(item["member"]).stem for item in report["stages"][kind]["extracted"]]
            images = [identifier.rsplit("_", 2)[0] for identifier in selected]
            report["sample_ids"] = selected
        write_json(root / "selection.json", report)
    paths = [
        str(root / item["member"])
        for stage in report["stages"].values()
        for item in stage["extracted"]
    ]
    inspect_files(paths, str(root / "file_inspection.json"))
    return report
