from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streaming_video_understanding.evaluation.metrics import summarize_records


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}, line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}, line {line_number}")
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize streaming evaluator JSONL records.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_records(_read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
    else:
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
