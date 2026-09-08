import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "download_data.py"
SPEC = importlib.util.spec_from_file_location("download_data", SCRIPT_PATH)
DOWNLOAD_DATA = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DOWNLOAD_DATA)


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str], status: int):
        self._body = body
        self.headers = headers
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size: int) -> bytes:
        body, self._body = self._body, b""
        return body


def test_download_archive_resumes_after_an_incomplete_response(tmp_path):
    archive = tmp_path / "archive.zip"
    requests = []
    responses = iter(
        [
            _FakeResponse(b"abc", {"Content-Length": "6"}, 200),
            _FakeResponse(
                b"def",
                {"Content-Length": "3", "Content-Range": "bytes 3-5/6"},
                206,
            ),
        ]
    )

    def opener(request, timeout):
        requests.append((request.headers.get("Range"), timeout))
        return next(responses)

    DOWNLOAD_DATA._download_archive(
        "https://example.invalid/archive.zip",
        archive,
        opener=opener,
        sleep=lambda _seconds: None,
    )

    assert archive.read_bytes() == b"abcdef"
    assert requests == [(None, 120), ("bytes=3-", 120)]
