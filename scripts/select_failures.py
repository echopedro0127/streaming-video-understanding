from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streaming_video_understanding.analysis.failures import select_failures, write_failure_review


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"failure input does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON at {path}, line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object at {path}, line {line_number}")
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Select deterministic streaming failure cases for review.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    for path in args.input:
        records.extend(_read_jsonl(path))
    selected = select_failures(records, limit=args.limit, seed=args.seed)
    paths = write_failure_review(selected, args.output_dir)
    print(f"selected={len(selected)} from input_records={len(records)}")
    for kind, path in paths.items():
        print(f"{kind}={path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
