"""Produce cluster uncertainty and blinded evidence queue from exact paired runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Share strict input validation with the existing paired-analysis CLI.
from analyze_policy_pairs import _load_jsonl
from streaming_video_understanding.analysis.pairs import analyze_pair_records
from streaming_video_understanding.analysis.allocation import allocation_diagnosis, evidence_review_queue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--baseline-policy", required=True)
    parser.add_argument("--candidate-input", required=True, type=Path)
    parser.add_argument("--candidate-policy", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-min", type=int)
    parser.add_argument("--video-max", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--review-limit", type=int, default=60)
    args = parser.parse_args()
    report_path = args.output_dir / "diagnosis.json"
    review_path = args.output_dir / "evidence-review.jsonl"
    if report_path.exists() or review_path.exists():
        parser.error("output artifacts already exist; choose a new output directory")
    try:
        pairs = analyze_pair_records(
            _load_jsonl(args.baseline_input), _load_jsonl(args.candidate_input),
            args.baseline_policy, args.candidate_policy,
            min_video_index=args.video_min, max_video_index=args.video_max,
        )
        report = allocation_diagnosis(pairs.rows, seed=args.seed, n_bootstrap=args.bootstrap)
        review = evidence_review_queue(pairs.rows, seed=args.seed, limit=args.review_limit)
    except ValueError as exc:
        parser.error(str(exc))
    report["baseline_policy"] = args.baseline_policy
    report["candidate_policy"] = args.candidate_policy
    report["evidence_review_sampling"] = {"method": "paired-outcome-balanced", "seed": args.seed,
                                         "n_rows": len(review), "status": "unannotated"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects hand-edited evidence annotations on rerun.
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with review_path.open("x", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in review)
    print(f"paired={report['n_pairs']} videos={report['n_videos']} delta={report['accuracy_delta']:.4f} report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
