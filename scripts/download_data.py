from __future__ import annotations

import argparse
import http.client
import re
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path


ARCHIVE_SIZE_GB = 9.08
MIN_FREE_GB = 25.0
ARCHIVE_URL = (
    "https://huggingface.co/datasets/mjuicem/StreamingBench/resolve/main/"
    "Real-Time%20Visual%20Understanding_1-50.zip?download=true"
)
DOWNLOAD_TIMEOUT_S = 120
DOWNLOAD_RETRIES = 8
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class _IncompleteDownloadError(Exception):
    pass


def _content_range_info(value: str | None) -> tuple[int, int, int | None] | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value.strip())
    if match is None:
        return None
    total = None if match.group(3) == "*" else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), total


def _download_archive(
    url: str,
    archive_path: Path,
    *,
    opener=urllib.request.urlopen,
    sleep=time.sleep,
) -> None:
    """Download a potentially large file, retaining partial data between retries."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(DOWNLOAD_RETRIES):
        existing = archive_path.stat().st_size if archive_path.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        request = urllib.request.Request(url, headers=headers)

        try:
            with opener(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                if existing and status != 206:
                    raise RuntimeError(
                        "The server did not honor the resume request "
                        f"(expected HTTP 206, got {status})."
                    )

                content_range = _content_range_info(response.headers.get("Content-Range"))
                if existing and (content_range is None or content_range[0] != existing):
                    raise RuntimeError(
                        "The server returned an invalid Content-Range for the partial archive."
                    )
                total = content_range[2] if content_range else None
                if total is None:
                    content_length = response.headers.get("Content-Length")
                    total = existing + int(content_length) if content_length else None
                if total is not None and existing > total:
                    raise RuntimeError(
                        f"Partial archive is larger than the server file ({existing} > {total} bytes)."
                    )

                mode = "ab" if existing else "wb"
                with archive_path.open(mode) as output:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        output.write(chunk)

                current = archive_path.stat().st_size
                if total is None or current == total:
                    return
                if current > total:
                    raise RuntimeError(
                        f"Downloaded more bytes than expected ({current} > {total} bytes)."
                    )
                raise _IncompleteDownloadError(
                    f"connection ended at {current} of {total} bytes"
                )
        except _IncompleteDownloadError as error:
            last_error = error
        except (http.client.IncompleteRead, TimeoutError, OSError) as error:
            last_error = error

        if attempt + 1 < DOWNLOAD_RETRIES:
            current = archive_path.stat().st_size if archive_path.exists() else 0
            delay = min(30, 2**attempt)
            print(
                f"Download interrupted at {current} bytes; retrying in {delay}s "
                f"({attempt + 2}/{DOWNLOAD_RETRIES})..."
            )
            sleep(delay)

    current = archive_path.stat().st_size if archive_path.exists() else 0
    raise RuntimeError(
        f"Unable to complete archive download after {DOWNLOAD_RETRIES} attempts; "
        f"partial file contains {current} bytes."
    ) from last_error


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"unsafe archive member path: {member.filename}")
        archive.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one bounded StreamingBench archive.")
    parser.add_argument(
        "--subset",
        choices=("real-time-1-50",),
        required=True,
        help="Only the first 50 real-time videos are supported in the MVP.",
    )
    parser.add_argument("--destination", type=Path, default=Path("data/streamingbench"))
    parser.add_argument(
        "--confirm-large-download",
        action="store_true",
        help="Confirm the approximately 9.08GB archive and extraction footprint.",
    )
    args = parser.parse_args()
    destination = args.destination.resolve()
    archive_path = destination / "Real-Time Visual Understanding_1-50.zip"
    extraction_dir = destination / "real-time-1-50"
    print(f"Archive target: {ARCHIVE_SIZE_GB:.2f}GB ({archive_path})")
    print(f"Extraction target: {extraction_dir}")
    if not args.confirm_large_download:
        raise SystemExit("Refusing large download: pass --confirm-large-download explicitly.")
    destination.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(destination).free / (1024**3)
    if free_gb < MIN_FREE_GB:
        raise SystemExit(
            f"Only {free_gb:.2f}GB is free at {destination}; at least {MIN_FREE_GB:.0f}GB is required."
        )
    print("Downloading from the official Hugging Face StreamingBench dataset repository...")
    _download_archive(ARCHIVE_URL, archive_path)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive_path, extraction_dir)
    print(f"Downloaded and extracted {args.subset} into {extraction_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
