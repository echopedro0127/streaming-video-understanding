"""Analyze semantic answer consistency under controlled option permutations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze_policy_pairs import _load_jsonl
from streaming_video_understanding.analysis.option_stability import analyze_option_stability
from streaming_video_understanding.evaluation.splits import load_split_sample_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, nargs="+", required=True)
    parser.add_argument("--permuted", type=Path, nargs="+", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["recent-8", "adaptive-semantic-v1"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.output_dir / "summary.json"
    rows_path = args.output_dir / "comparisons.jsonl"
    if summary_path.exists() or rows_path.exists():
        parser.error("output already exists; choose a new output directory")
    try:
        result = analyze_option_stability(
            [row for path in args.original for row in _load_jsonl(path)],
            [row for path in args.permuted for row in _load_jsonl(path)],
            sample_ids=sorted(load_split_sample_ids(args.split)), policies=args.policies, seeds=args.seeds,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result["summary"], indent=2, sort_keys=True, allow_nan=False) + "\n")
    with rows_path.open("x", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in result["rows"])
    print(f"comparisons={len(result['rows'])} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
