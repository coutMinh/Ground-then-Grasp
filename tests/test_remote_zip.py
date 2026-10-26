import io
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from g2g.data.download import read_indexed_member
from g2g.data.remote_zip import HTTPRangeReader, RangeReadError


@pytest.fixture
def archive_server():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with archive.open("nested/fixture.txt", "w", force_zip64=True) as member:
            member.write(b"verified fixture bytes" * 100)
    data = stream.getvalue()
    midpoint = len(data) // 2
    parts = {"/aa": data[:midpoint], "/ab": data[midpoint:], "/ignored": data}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = parts[self.path]
            if self.path == "/ignored":
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            end = min(end, len(payload) - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self.wfile.write(payload[start : end + 1])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_zip64_member_across_concatenated_parts_and_cache(archive_server):
    with HTTPRangeReader(
        [archive_server + "/aa", archive_server + "/ab"], block_size=23, cache_blocks=2
    ) as reader:
        with zipfile.ZipFile(reader) as archive:
            assert archive.namelist() == ["nested/fixture.txt"]
            assert archive.read("nested/fixture.txt") == b"verified fixture bytes" * 100
            info = archive.getinfo("nested/fixture.txt")
        assert read_indexed_member(reader, info) == b"verified fixture bytes" * 100
        reader.seek(reader.size + 10)
        assert reader.read() == b""
        assert len(reader._cache) <= 2


def test_refuses_server_that_ignores_range(archive_server):
    with pytest.raises(RangeReadError, match="not honored"):
        HTTPRangeReader([archive_server + "/ignored"])


def test_multiple_readers_can_share_disk_cache(archive_server, tmp_path):
    def read_fixture(_):
        with (
            HTTPRangeReader(
                [archive_server + "/aa", archive_server + "/ab"], block_size=23, cache_dir=tmp_path
            ) as reader,
            zipfile.ZipFile(reader) as archive,
        ):
            return archive.read("nested/fixture.txt")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read_fixture, range(8)))
    assert results == [b"verified fixture bytes" * 100] * 8
