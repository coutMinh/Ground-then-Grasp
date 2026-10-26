"""Explicit archive inventory/extraction, independent of native dataset schema."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

from g2g.data.remote_zip import HTTPRangeReader
from g2g.utils.io import sha256_file, write_json


def inspect_archive(
    urls: list[str],
    output: str,
    *,
    members: list[str] | None = None,
    destination: str | None = None,
    limit: int = 20,
) -> dict:
    if limit < 1:
        raise ValueError("Inventory preview limit must be positive.")
    report = {"member_count": 0, "preview": [], "extracted": []}
    with (
        HTTPRangeReader(
            urls,
            token=os.environ.get("HF_TOKEN"),
            block_size=4 * 1024 * 1024,
            cache_dir=Path(output).parent / "http-cache",
        ) as reader,
        zipfile.ZipFile(reader) as archive,
    ):
        infos = archive.infolist()
        report["member_count"] = len(infos)
        report["preview"] = [
            {"name": info.filename, "size": info.file_size, "crc32": f"{info.CRC:08x}"}
            for info in infos[:limit]
        ]
        for name in members or []:
            if destination is None:
                raise ValueError("Extraction requires --destination.")
            root = Path(destination).resolve()
            target = (root / name).resolve()
            if not target.is_relative_to(root) or target == root:
                raise ValueError("Archive member escapes the destination directory.")
            info = archive.getinfo(name)
            if info.is_dir() or info.file_size > 100 * 1024 * 1024:
                raise ValueError("Select a regular sample member no larger than 100 MiB.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink)
            report["extracted"].append({"member": name, "sha256": sha256_file(target)})
    write_json(output, report)
    return report
