from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


ARCHIVE_SIZE_GB = 9.08
MIN_FREE_GB = 25.0
ARCHIVE_URL = (
    "https://huggingface.co/datasets/mjuicem/StreamingBench/resolve/main/"
    "Real-Time%20Visual%20Understanding_1-50.zip?download=true"
)


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
    urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive_path, extraction_dir)
    print(f"Downloaded and extracted {args.subset} into {extraction_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
