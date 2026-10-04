from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PIL import Image

from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
from streaming_video_understanding.data.streamingbench import load_streamingbench_file
from streaming_video_understanding.evaluation.engine import evaluate_sample
from streaming_video_understanding.evaluation.failures import FailureStore
from streaming_video_understanding.evaluation.permutation import permute_sample
from streaming_video_understanding.evaluation.profiling import RuntimeProfileStore
from streaming_video_understanding.evaluation.resume import ResumeStore
from streaming_video_understanding.evaluation.splits import load_split_sample_ids
from streaming_video_understanding.models.mock import MockVLMBackend
from streaming_video_understanding.models.smolvlm import SmolVLMBackend
from streaming_video_understanding.utils.config import load_config
from streaming_video_understanding.utils.config import configuration_id
from streaming_video_understanding.utils.seed import set_seed


_SEMANTIC_POLICY_PATTERN = re.compile(
    r"^recent-\d+\+(?:option-)?semantic(?:-diverse)?-memory-\d+$",
    re.IGNORECASE,
)


def _requires_semantic_scorer(policy_ids: tuple[str, ...]) -> bool:
    return any(
        _SEMANTIC_POLICY_PATTERN.fullmatch(policy_id.strip())
        for policy_id in policy_ids
    )


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


def _load_samples(
    data_root: Path | None,
    *,
    annotation_path: Path | None,
    sample_fps: float,
    cache_decoded_frames: bool = False,
) -> tuple[list[StreamingSample], object]:
    if data_root is None:
        return _synthetic_samples(), SyntheticFrameReader()
    supported = {".csv", ".json", ".jsonl", ".parquet"}
    if annotation_path is None:
        candidates = sorted(
            path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in supported
        )
        if not candidates:
            raise FileNotFoundError(f"No annotation file found below {data_root}")
        if len(candidates) > 1:
            raise FileNotFoundError(
                "Multiple annotation files found; pass --annotations explicitly: "
                + ", ".join(str(path) for path in candidates)
            )
        annotation_path = candidates[0]
    elif not annotation_path.is_absolute():
        annotation_path = (data_root / annotation_path).resolve()
    rows = load_streamingbench_file(annotation_path)
    resolved = [
        replace(
            row,
            video_path=(
                (annotation_path.parent / row.video_path).resolve()
                if not row.video_path.is_absolute()
                and (annotation_path.parent / row.video_path).exists()
                else (data_root / row.video_path).resolve()
                if not row.video_path.is_absolute()
                else row.video_path
            ),
        )
        for row in rows
    ]
    from streaming_video_understanding.data.video import CachedFrameReader, FrameReader

    reader = FrameReader(sample_fps=sample_fps)
    return resolved, CachedFrameReader(reader) if cache_decoded_frames else reader


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the causal streaming video evaluator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--backend", choices=("mock", "smolvlm"), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--option-permutation-seed", type=int, default=None)
    parser.add_argument("--profile-output", type=Path, default=None)
    parser.add_argument(
        "--cache-decoded-frames",
        action="store_true",
        help="Reuse a bounded decoded query across policies for the same sample.",
    )
    parser.add_argument("--policy", default=None)
    parser.add_argument(
        "--all-policies",
        action="store_true",
        help="Run every policy_id declared in the YAML config.",
    )
    parser.add_argument("--mock-answer", choices=("A", "B", "C", "D"), default="A")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.policy and args.all_policies:
        raise SystemExit("--policy and --all-policies are mutually exclusive")
    config = load_config(args.config)
    backend_name = args.backend or config.model.backend
    failure_path = args.output.with_name(args.output.stem + "-failures.jsonl")
    profile_store = RuntimeProfileStore(args.profile_output) if args.profile_output else None
    if args.output.exists():
        if args.overwrite:
            args.output.unlink()
            if failure_path.exists():
                failure_path.unlink()
            if args.profile_output and args.profile_output.exists():
                args.profile_output.unlink()
        elif not args.resume:
            raise SystemExit("Output exists; use --resume or --overwrite explicitly")

    set_seed(config.experiment.seed)
    samples, frame_reader = _load_samples(
        args.data_root,
        annotation_path=args.annotations,
        sample_fps=config.runtime.sample_fps,
        cache_decoded_frames=args.cache_decoded_frames,
    )
    if args.split is not None:
        split_path = args.split if args.split.is_absolute() else (PROJECT_ROOT / args.split).resolve()
        split_ids = load_split_sample_ids(split_path)
        samples = [sample for sample in samples if sample.sample_id in split_ids]
        if not samples:
            raise SystemExit(f"--split selected no samples from {split_path}")
    if args.limit is not None:
        samples = samples[: args.limit]
    if args.all_policies:
        policy_ids = config.experiment.policy_ids
        if not policy_ids:
            raise SystemExit("--all-policies requires non-empty policy_ids in the config")
    else:
        policy_ids = (args.policy or f"recent-{config.experiment.visual_budget}",)
    semantic_scorer = None
    if _requires_semantic_scorer(policy_ids):
        from streaming_video_understanding.retrieval.clip import ClipFrameScorer

        semantic_scorer = ClipFrameScorer(
            model_id=config.retrieval.model_id,
            device=config.retrieval.device,
            cache_dir=config.retrieval.cache_dir,
            image_batch_size=config.retrieval.image_batch_size,
        )
    store = ResumeStore(args.output)
    failure_store = FailureStore(failure_path)
    backend = (
        MockVLMBackend(args.mock_answer)
        if backend_name == "mock"
        else SmolVLMBackend(
            model_id=config.model.model_id,
            max_new_tokens=config.model.max_new_tokens,
        )
    )

    processed = 0
    skipped = 0
    for sample in samples:
        for policy_id in policy_ids:
            evaluation_sample = sample
            record_metadata: dict[str, object] = {}
            permutation_suffix = ""
            if args.option_permutation_seed is not None:
                permutation = permute_sample(sample, seed=args.option_permutation_seed)
                evaluation_sample = permutation.sample
                permutation_suffix = f"-option-permutation-{args.option_permutation_seed}"
                record_metadata = {
                    "option_permutation": list(permutation.permutation),
                    "original_options": list(sample.options),
                    "permuted_options": list(permutation.sample.options),
                    "original_ground_truth": permutation.original_answer,
                    "option_permutation_seed": args.option_permutation_seed,
                }
            run_config_id = configuration_id(
                config,
                policy_id=policy_id,
                backend=backend_name,
            ) + permutation_suffix
            if store.is_complete(sample.sample_id, run_config_id):
                skipped += 1
                continue
            try:
                evaluate_sample(
                    evaluation_sample,
                    config=config,
                    frame_reader=frame_reader,
                    backend=backend,
                    output_store=store,
                    policy_id=policy_id,
                    failure_store=failure_store,
                    config_id=run_config_id,
                    record_metadata=record_metadata,
                    profile_store=profile_store,
                    semantic_scorer=semantic_scorer,
                )
                processed += 1
            except Exception as exc:
                failure_store.append_failure(
                    {
                        "sample_id": sample.sample_id,
                        "video_id": sample.video_id,
                        "video_path": str(sample.video_path),
                        "config_id": run_config_id,
                        "policy_id": policy_id,
                        "question": sample.question,
                        "query_time_s": sample.query_time_s,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"failed sample={sample.sample_id} policy={policy_id}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
    print(f"processed={processed} skipped={skipped} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
