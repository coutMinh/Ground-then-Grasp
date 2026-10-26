"""Explicit, image-disjoint manifests from actual labels and official object IDs."""

from __future__ import annotations

import csv
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from g2g.data.remote_zip import HTTPRangeReader
from g2g.data.schema import SampleKey
from g2g.inspection.samples import GA_REVISION, GAPP_REVISION, archive_urls
from g2g.inspection.upstream import read_split
from g2g.utils.io import sha256_file, write_json

SPLITS = ("train", "val", "test_seen", "test_unseen")
FIELDS = [
    "sample_id",
    "image_id",
    "object_id",
    "image_member",
    "label_member",
    "mask_member",
    "instruction_member",
    "split",
]


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected manifest columns: {path}")
        rows = list(reader)
    identifiers = set()
    images = defaultdict(set)
    for row in rows:
        key = SampleKey.parse(row["sample_id"])
        if row["sample_id"] in identifiers:
            raise ValueError("Duplicate sample IDs in manifest.")
        identifiers.add(row["sample_id"])
        if row["image_id"] != key.image_id or row["object_id"] != key.object_id:
            raise ValueError("Manifest IDs disagree with the sample filename.")
        if row["split"] not in SPLITS:
            raise ValueError("Unknown split.")
        for field, member in key.members().items():
            if row[f"{field}_member"] != member:
                raise ValueError("Manifest member disagrees with the verified filename convention.")
        images[key.image_id].add(row["split"])
    if any(len(splits) > 1 for splits in images.values()):
        raise ValueError("Image leakage across splits.")
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_manifests(
    upstream: str,
    output: str,
    cache_dir: str,
    *,
    train_images: int = 100,
    val_images: int = 20,
    test_images: int = 20,
    seed: int = 42,
) -> dict:
    if train_images < 1 or val_images < 1 or test_images < 0:
        raise ValueError(
            "Train/val counts must be positive; test count 0 means full official lists."
        )
    split_root = Path(upstream) / "split/grasp-anything++"
    official = {
        "train": set(read_split(split_root / "train/seen.obj")),
        "test_seen": set(read_split(split_root / "test/seen.obj")),
        "test_unseen": set(read_split(split_root / "test/unseen.obj")),
    }
    test_image_ids = {
        entry.rsplit("_", 1)[0]
        for split in ("test_seen", "test_unseen")
        for entry in official[split]
    }
    allowed = set.union(*official.values())
    groups = {split: defaultdict(list) for split in official}
    print("Indexing actual part labels for official object IDs...", flush=True)
    with (
        HTTPRangeReader(
            archive_urls("grasp_label_positive"),
            block_size=64 * 1024,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),
        ) as reader,
        zipfile.ZipFile(reader) as archive,
    ):
        for info in archive.infolist():
            if not info.filename.startswith("grasp_label_positive/") or not info.filename.endswith(
                ".pt"
            ):
                continue
            stem = info.filename.removeprefix("grasp_label_positive/")[:-3]
            object_id = stem.rsplit("_", 1)[0]
            if object_id not in allowed:
                continue
            key = SampleKey.parse(stem)
            for split, objects in official.items():
                if object_id in objects and not (
                    split == "train" and key.image_id in test_image_ids
                ):
                    groups[split][key.image_id].append(key)
    randomizer = random.Random(seed)
    train_pool = sorted(groups["train"])
    randomizer.shuffle(train_pool)
    if train_images + val_images > len(train_pool):
        raise ValueError(
            f"Only {len(train_pool)} eligible images; requested train+val is too large."
        )
    selected = {
        "train": train_pool[:train_images],
        "val": train_pool[train_images : train_images + val_images],
    }
    for split in ("test_seen", "test_unseen"):
        images = sorted(groups[split])
        randomizer.shuffle(images)
        selected[split] = images[:test_images] if test_images else images
    destination = Path(output)
    report = {
        "seed": seed,
        "ga_revision": GA_REVISION,
        "gapp_revision": GAPP_REVISION,
        "protocol": (
            "official object lists; actual part entries; all test images excluded from train pool"
        ),
        "union_policy": "all parts of permitted train objects only; one training item per image",
        "instruction_source": "native instruction_member pickle string, not duplicated in CSV",
        "available_train_pool_images": len(train_pool),
        "splits": {},
    }
    all_rows = []
    for split in SPLITS:
        source = "train" if split == "val" else split
        rows = []
        for image_id in selected[split]:
            for key in sorted(groups[source][image_id], key=lambda item: item.sample_id):
                members = key.members()
                rows.append(
                    {
                        "sample_id": key.sample_id,
                        "image_id": key.image_id,
                        "object_id": key.object_id,
                        "split": split,
                        **{f"{field}_member": value for field, value in members.items()},
                    }
                )
        path = destination / f"{split}.csv"
        write_manifest(path, rows)
        read_manifest(path)
        all_rows.extend(rows)
        report["splits"][split] = {
            "rows": len(rows),
            "images": len(selected[split]),
            "sha256": sha256_file(path),
        }
    combined = destination / "all.csv"
    write_manifest(combined, all_rows)
    read_manifest(combined)  # Cross-file image leakage check, not just per-file checks.
    report["all_sha256"] = sha256_file(combined)
    write_json(destination / "protocol.json", report)
    return report
