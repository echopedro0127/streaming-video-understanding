from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


DATASET = "mjuicem/StreamingBench"
CONFIG = "Real_Time_Visual_Understanding"
SPLIT = "Real_Time_Visual_Understanding"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
REQUEST_TIMEOUT_S = 60
QUESTION_ID_PATTERN = re.compile(r"_sample_(\d+)_\d+$")


def _map_row(row: dict[str, object], *, max_sample: int) -> dict[str, object] | None:
    question_id = str(row.get("question_id", "")).strip()
    match = QUESTION_ID_PATTERN.search(question_id)
    if match is None:
        raise ValueError(f"Cannot infer sample number from question_id: {question_id!r}")
    sample_number = int(match.group(1))
    if sample_number < 1 or sample_number > max_sample:
        return None
    mapped = dict(row)
    mapped["video"] = f"sample_{sample_number}/video.mp4"
    return mapped


def _fetch_rows(*, opener=urllib.request.urlopen):
    offset = 0
    total_rows: int | None = None
    while total_rows is None or offset < total_rows:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        url = f"{ROWS_API}?{query}"
        try:
            with opener(url, timeout=REQUEST_TIMEOUT_S) as response:
                payload = json.load(response)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to fetch StreamingBench annotations at offset {offset}") from exc

        total_rows = int(payload.get("num_rows_total", 0))
        page = payload.get("rows", [])
        if not isinstance(page, list) or not page:
            break
        for item in page:
            if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
                raise ValueError(f"Unexpected annotation API row at offset {offset}: {item!r}")
            yield item["row"]
        offset += len(page)


def download_annotations(destination: Path, *, max_sample: int = 50) -> int:
    if max_sample < 1:
        raise ValueError("max_sample must be positive")
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in _fetch_rows():
        mapped = _map_row(row, max_sample=max_sample)
        if mapped is None:
            continue
        question_id = str(mapped["question_id"])
        if question_id in seen:
            continue
        seen.add(question_id)
        selected.append(mapped)

    if not selected:
        raise RuntimeError("The annotation API returned no rows for the requested subset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download bounded StreamingBench QA annotations.")
    parser.add_argument(
        "--subset",
        choices=("real-time-1-50",),
        default="real-time-1-50",
        help="The local video subset whose QA rows should be retained.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/streamingbench/real-time-1-50/annotations.jsonl"),
    )
    args = parser.parse_args()
    max_sample = 50 if args.subset == "real-time-1-50" else 0
    count = download_annotations(args.destination, max_sample=max_sample)
    print(f"wrote {count} annotations to {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
