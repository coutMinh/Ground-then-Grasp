"""Seekable HTTP Range reader for ZIP/ZIP64 and concatenated archive parts.

Original resolve URLs are used on each request so expiring signed redirects
can be refreshed. No archive is silently downloaded when Range is unsupported.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import threading
from bisect import bisect_right
from collections import OrderedDict
from itertools import accumulate
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RangeReadError(OSError):
    """HTTP server did not honor a bounded byte-range request."""


def _persist_cache(path: Path, data: bytes) -> None:
    """Publish immutable cache blocks safely with concurrent Windows readers."""
    if path.exists() and path.stat().st_size == len(data):
        return
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_bytes(data)
    try:
        temporary.replace(path)
    except PermissionError:
        # Windows cannot replace a file another worker is reading. A complete
        # same-key block means a competing worker has already published it.
        if not path.exists() or path.stat().st_size != len(data):
            raise
    finally:
        temporary.unlink(missing_ok=True)


class HTTPRangeReader(io.RawIOBase):
    def __init__(
        self,
        urls: list[str],
        *,
        block_size: int = 1024 * 1024,
        cache_blocks: int = 8,
        timeout: float = 30,
        token: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        super().__init__()
        if not urls or block_size < 1 or cache_blocks < 1 or timeout <= 0:
            raise ValueError("URLs, block/cache sizes and timeout must be valid.")
        self.urls = list(urls)
        for url in self.urls:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
                raise ValueError("Archive URLs must be HTTP(S) without embedded credentials.")
        self.block_size, self.cache_blocks, self.timeout = block_size, cache_blocks, timeout
        self._token = token
        self._disk_cache = Path(cache_dir) if cache_dir is not None else None
        if self._disk_cache:
            self._disk_cache.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        for scheme in ("https://", "http://"):
            self._session.mount(scheme, HTTPAdapter(max_retries=retries))
        self._position = 0
        self._cache: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        try:
            self.sizes = [self._request(index, 0, 0)[1] for index in range(len(urls))]
        except BaseException:
            self.close()
            raise
        self.size = sum(self.sizes)
        self._ends = list(accumulate(self.sizes))

    def _request(self, index: int, start: int, end: int) -> tuple[bytes, int]:
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        # Never forward a HF token to an arbitrary URL passed to this generic reader.
        if self._token and urlparse(self.urls[index]).hostname == "huggingface.co":
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            with self._session.get(
                self.urls[index], headers=headers, stream=True, timeout=self.timeout
            ) as response:
                if response.status_code != 206:
                    raise RangeReadError(f"Range not honored (HTTP {response.status_code}).")
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                )
                if not match:
                    raise RangeReadError("Missing or invalid Content-Range.")
                actual_start, actual_end, total = map(int, match.groups())
                if actual_start != start or actual_end != min(end, total - 1):
                    raise RangeReadError("Server returned an unexpected byte range.")
                expected = actual_end - actual_start + 1
                data = response.raw.read(expected + 1)
                if len(data) != expected:
                    raise RangeReadError("Truncated or oversized byte-range response.")
                return data, total
        except requests.RequestException:
            raise RangeReadError("Range request failed; URL/credentials omitted.") from None

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self._checkClosed()
        if whence not in {os.SEEK_SET, os.SEEK_CUR, os.SEEK_END}:
            raise ValueError("Invalid seek origin.")
        position = (
            offset + {os.SEEK_SET: 0, os.SEEK_CUR: self._position, os.SEEK_END: self.size}[whence]
        )
        if position < 0:
            raise ValueError("Cannot seek before the start of an archive.")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        remaining = max(0, self.size - self._position)
        remaining = remaining if size is None or size < 0 else min(size, remaining)
        output = bytearray()
        while remaining:
            index = bisect_right(self._ends, self._position)
            base = self._ends[index - 1] if index else 0
            part_size = self.sizes[index]
            local = self._position - base
            # zipfile reads the whole central directory in one call. Do not
            # turn that sequential request into hundreds of small HTTP round trips.
            count = min(remaining, part_size - local)
            if count >= 2 * self.block_size:
                fingerprint = hashlib.sha256(
                    f"large:{self.urls[index]}:{part_size}:{local}:{count}".encode()
                ).hexdigest()
                cached = self._disk_cache / fingerprint if self._disk_cache else None
                if cached is not None and cached.exists() and cached.stat().st_size == count:
                    data = cached.read_bytes()
                else:
                    data, total = self._request(index, local, local + count - 1)
                    if total != part_size:
                        raise RangeReadError("Archive changed size; pin its revision.")
                    if cached is not None:
                        _persist_cache(cached, data)
                self._position += count
                remaining -= count
                if not output and not remaining:
                    return data
                output.extend(data)
                continue
            block = local // self.block_size
            key = (index, block)
            if key not in self._cache:
                start = block * self.block_size
                end = min(start + self.block_size, part_size) - 1
                fingerprint = hashlib.sha256(
                    f"{self.urls[index]}:{part_size}:{start}:{end}".encode()
                ).hexdigest()
                cached = self._disk_cache / fingerprint if self._disk_cache else None
                if (
                    cached is not None
                    and cached.exists()
                    and cached.stat().st_size == end - start + 1
                ):
                    data = cached.read_bytes()
                else:
                    data, total = self._request(index, start, end)
                    if total != part_size:
                        raise RangeReadError(
                            "Archive changed size during reading; pin its revision."
                        )
                    if cached is not None:
                        _persist_cache(cached, data)
                self._cache[key] = data
                if len(self._cache) > self.cache_blocks:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            block_data = self._cache[key]
            within = local % self.block_size
            count = min(remaining, len(block_data) - within)
            output.extend(block_data[within : within + count])
            self._position += count
            remaining -= count
        return bytes(output)

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        if hasattr(self, "_session"):
            self._session.close()
        if hasattr(self, "_cache"):
            self._cache.clear()
        super().close()
