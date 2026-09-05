from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PIL import Image

from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
from streaming_video_understanding.data.streamingbench import load_streamingbench_csv
from streaming_video_understanding.evaluation.engine import evaluate_sample
from streaming_video_understanding.evaluation.failures import FailureStore
from streaming_video_understanding.evaluation.resume import ResumeStore
from streaming_video_understanding.models.mock import MockVLMBackend
from streaming_video_understanding.models.smolvlm import SmolVLMBackend
from streaming_video_understanding.utils.config import load_config


class SyntheticFrameReader:
    def read_until(self, video_path: Path, end_time_s: float):
        for frame_id in range(int(end_time_s) + 1):
            yield Image.new("RGB", (32, 32), color=(40 + frame_id % 180, 80, 120)), FrameRecord(
                frame_id=frame_id,
                timestamp_s=float(frame_id),
            )


def _synthetic_samples() -> list[StreamingSample]:
    return [
        StreamingSample(
            sample_id=f"synthetic-{index}",
            video_id=f"synthetic-video-{index}",
            video_path=Path(f"synthetic-{index}.mp4"),
            question="Which option describes the visible stream?",
            options=("a synthetic stream", "a dog", "a car", "a book"),
            answer="A",
            query_time_s=float(5 + index * 5),
            task_type="Synthetic smoke",
            temporal_clue_type="Prior",
        )
        for index in range(3)
    ]


def _load_samples(data_root: Path | None) -> tuple[list[StreamingSample], object]:
    if data_root is None:
        return _synthetic_samples(), SyntheticFrameReader()
    csv_paths = sorted(data_root.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV benchmark file found below {data_root}")
    rows = load_streamingbench_csv(csv_paths[0])
    resolved = [
        replace(
            row,
            video_path=(data_root / row.video_path).resolve()
            if not row.video_path.is_absolute()
            else row.video_path,
        )
        for row in rows
    ]
    from streaming_video_understanding.data.video import FrameReader

    return resolved, FrameReader(sample_fps=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the causal streaming video evaluator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--backend", choices=("mock", "smolvlm"), default="mock")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--mock-answer", choices=("A", "B", "C", "D"), default="A")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.output.exists():
        if args.overwrite:
            args.output.unlink()
        elif not args.resume:
            raise SystemExit("Output exists; use --resume or --overwrite explicitly")

    config = load_config(args.config)
    samples, frame_reader = _load_samples(args.data_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    policy_id = args.policy or f"recent-{config.experiment.visual_budget}"
    store = ResumeStore(args.output) if args.resume else ResumeStore(args.output)
    failure_store = FailureStore(args.output.with_name(args.output.stem + "-failures.jsonl"))
    backend = MockVLMBackend(args.mock_answer) if args.backend == "mock" else SmolVLMBackend()

    processed = 0
    skipped = 0
    for sample in samples:
        if store.is_complete(sample.sample_id, policy_id):
            skipped += 1
            continue
        evaluate_sample(
            sample,
            config=config,
            frame_reader=frame_reader,
            backend=backend,
            output_store=store,
            policy_id=policy_id,
            failure_store=failure_store,
        )
        processed += 1
    print(f"processed={processed} skipped={skipped} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
