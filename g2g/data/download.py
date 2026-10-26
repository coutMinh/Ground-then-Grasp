"""Parallel subset extraction with one central-directory parse per archive.

Workers read individual local ZIP headers and compressed member bytes by offset;
they do not each hold a multi-million-entry ZipFile index in memory.
"""

from __future__ import annotations

import os
import struct
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from g2g.data.manifest import read_manifest
from g2g.data.remote_zip import HTTPRangeReader
from g2g.inspection.samples import archive_urls
from g2g.utils.io import sha256_file, write_json


def read_indexed_member(reader: HTTPRangeReader, info: zipfile.ZipInfo) -> bytes:
    if info.flag_bits & 1:
        raise ValueError("Encrypted members are unsupported.")
    reader.seek(info.header_offset)
    header = reader.read(zipfile.sizeFileHeader)
    if len(header) != zipfile.sizeFileHeader:
        raise ValueError("Truncated ZIP local header.")
    fields = struct.unpack(zipfile.structFileHeader, header)
    if fields[0] != zipfile.stringFileHeader:
        raise ValueError("Invalid ZIP local header signature.")
    name_size, extra_size = fields[-2:]
    name = reader.read(name_size)
    encoding = "utf-8" if info.flag_bits & 0x800 else "cp437"
    if name.decode(encoding) != info.filename:
        raise ValueError("Local ZIP filename differs from the index.")
    reader.seek(extra_size, os.SEEK_CUR)
    with zipfile.ZipExtFile(reader, "r", info) as member:
        return member.read()  # validates CRC against central-directory metadata


def _selected_infos(kind: str, names: set[str], cache_dir: Path) -> list[zipfile.ZipInfo]:
    print(f"Indexing {kind}: {len(names)} requested members...", flush=True)
    with (
        HTTPRangeReader(
            archive_urls(kind),
            block_size=64 * 1024,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),
        ) as reader,
        zipfile.ZipFile(reader) as archive,
    ):
        return [archive.getinfo(name) for name in sorted(names)]


def _extract_kind(
    kind: str, infos: list[zipfile.ZipInfo], root: Path, cache_dir: Path, workers: int
) -> dict:
    local = threading.local()
    readers = []
    lock = threading.Lock()

    def extract(info: zipfile.ZipInfo) -> dict:
        target = (root / info.filename).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Member escapes data root.")
        if target.exists():
            content = target.read_bytes()
            if len(content) == info.file_size and zlib.crc32(content) == info.CRC:
                return {"member": info.filename, "sha256": sha256_file(target), "cached": True}
        if not hasattr(local, "reader"):
            local.reader = HTTPRangeReader(
                archive_urls(kind),
                block_size=64 * 1024,
                cache_dir=cache_dir,
                token=os.environ.get("HF_TOKEN"),
            )
            with lock:
                readers.append(local.reader)
        content = read_indexed_member(local.reader, info)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
        return {"member": info.filename, "sha256": sha256_file(target), "cached": False}

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            completed = list(pool.map(extract, infos))
    finally:
        for reader in readers:
            reader.close()
    print(f"Verified {len(completed)} {kind} members.", flush=True)
    return {"members": completed, "count": len(completed)}


def download_subset(manifest: str, data_root: str, cache_dir: str, workers: int = 8) -> dict:
    if workers < 1 or workers > 32:
        raise ValueError("Use between 1 and 32 download workers.")
    rows = read_manifest(manifest)
    root = Path(data_root).resolve()
    fields = {
        "grasp_label_positive": "label_member",
        "grasp_instructions": "instruction_member",
        "part_mask": "mask_member",
        "images": "image_member",
    }
    report = {"manifest_sha256": sha256_file(manifest), "stages": {}, "complete": False}
    try:
        for kind, field in fields.items():
            names = {row[field] for row in rows}
            infos = _selected_infos(kind, names, Path(cache_dir))
            report["stages"][kind] = _extract_kind(kind, infos, root, Path(cache_dir), workers)
            write_json(root / "download_report.json", report)
        report["complete"] = True
    except Exception as error:
        report["error_type"] = type(error).__name__
        write_json(root / "download_report.json", report)
        raise
    write_json(root / "download_report.json", report)
    return report
