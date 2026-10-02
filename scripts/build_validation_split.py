from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streaming_video_understanding.data.streamingbench import load_streamingbench_file
from streaming_video_understanding.evaluation.splits import (
    build_deterministic_split,
    write_split_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic StreamingBench split.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_streamingbench_file(args.annotations)
    split = build_deterministic_split(
        samples,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    paths = write_split_files(split, args.output_dir)
    print(
        f"validation={len(split.validation)} final_test={len(split.final_test)} "
        f"seed={args.seed} output={args.output_dir}"
    )
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
