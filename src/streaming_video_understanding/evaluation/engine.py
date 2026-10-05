from __future__ import annotations

import re
import time
import gc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..data.schema import FrameRecord, StreamingSample
from ..models.base import ModelBackend, ModelRequest
from ..models.prompt import build_prompt
from ..retrieval.base import OptionAwareFrameScorer, SemanticFrameScorer
from ..retrieval.option import OptionScoreItem, aggregate_option_scores
from ..streaming.buffer import CausalFrameBuffer
from ..streaming.memory import BoundedTemporalMemory
from ..streaming.sampler import (
    recent_window,
    timestamp_aware_history,
    uniform_history,
)
from ..streaming.semantic import select_semantic_history
from ..utils.config import ProjectConfig
from .failures import FailureStore
from .profiling import RuntimeProfileStore
from .resume import ResumeStore


def _peak_gpu_memory_mb() -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def _reset_gpu_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _policy_parts(policy_id: str, budget: int) -> tuple[str, int, int]:
    normalized = policy_id.lower().strip()
    if normalized in {"recent", "recent-window"}:
        return "recent", budget, 0
    if normalized in {"uniform", "uniform-history"}:
        return "uniform", 0, budget
    match = re.fullmatch(
        r"recent-(\d+)\+(uniform|timestamp-aware|semantic|semantic-diverse|"
        r"option-semantic|option-semantic-diverse)-memory-(\d+)",
        normalized,
    )
    if match:
        policy_kind = match.group(2)
        recent_capacity = int(match.group(1))
        memory_capacity = int(match.group(3))
    else:
        match = re.fullmatch(r"(recent|uniform)-(\d+)", normalized)
        if match:
            if match.group(1) == "recent":
                policy_kind, recent_capacity, memory_capacity = (
                    "recent",
                    int(match.group(2)),
                    0,
                )
            else:
                policy_kind, recent_capacity, memory_capacity = (
                    "uniform",
                    0,
                    int(match.group(2)),
                )
        else:
            raise ValueError(
                f"Unsupported policy_id {policy_id!r}; expected recent-B, uniform-B, "
                "or recent-K+uniform-memory-M/timestamp-aware-memory-M/"
                "semantic-memory-M/semantic-diverse-memory-M/"
                "option-semantic-memory-M/option-semantic-diverse-memory-M"
            )
    requested_capacity = recent_capacity + memory_capacity
    if requested_capacity > budget:
        raise ValueError(
            f"policy {policy_id!r} requests {requested_capacity} frames, "
            f"exceeding visual_budget {budget}"
        )
    return policy_kind, recent_capacity, memory_capacity


def _select_policy(
    frames: Sequence[FrameRecord],
    *,
    query_time_s: float,
    budget: int,
    policy_id: str,
    tolerance_s: float,
) -> tuple[tuple[FrameRecord, ...], tuple[FrameRecord, ...], bool]:
    policy_kind, recent_capacity, memory_capacity = _policy_parts(policy_id, budget)
    if memory_capacity == 0:
        selected = recent_window(
            frames,
            visible_until_s=query_time_s,
            capacity=recent_capacity,
            tolerance_s=tolerance_s,
        )
        return selected, (), False
    if recent_capacity == 0:
        selected = uniform_history(
            frames,
            visible_until_s=query_time_s,
            capacity=memory_capacity,
            tolerance_s=tolerance_s,
        )
        return selected, selected, policy_kind == "timestamp-aware"

    ordered = tuple(sorted(frames, key=lambda frame: (frame.timestamp_s, frame.frame_id)))
    recent = recent_window(
        ordered,
        visible_until_s=query_time_s,
        capacity=recent_capacity,
        tolerance_s=tolerance_s,
    )
    recent_ids = {frame.frame_id for frame in recent}
    history = tuple(frame for frame in ordered if frame.frame_id not in recent_ids)
    history_sampler = (
        timestamp_aware_history if policy_kind == "timestamp-aware" else uniform_history
    )
    memory = history_sampler(
        history,
        visible_until_s=query_time_s,
        capacity=memory_capacity,
        tolerance_s=tolerance_s,
    )
    selected = tuple(sorted((*memory, *recent), key=lambda frame: (frame.timestamp_s, frame.frame_id)))
    return selected, memory, policy_kind == "timestamp-aware"


def _select_semantic_policy(
    frames: Sequence[FrameRecord],
    *,
    video_path: Path,
    images_by_id: Mapping[int, Any],
    query: str,
    query_time_s: float,
    budget: int,
    policy_id: str,
    tolerance_s: float,
    semantic_scorer: SemanticFrameScorer | None,
    candidate_multiplier: int,
    cluster_window_s: float,
    diversity_weight: float,
) -> tuple[
    tuple[FrameRecord, ...],
    tuple[FrameRecord, ...],
    tuple[dict[str, float | int], ...],
    float,
    float,
]:
    policy_kind, recent_capacity, memory_capacity = _policy_parts(policy_id, budget)
    if semantic_scorer is None:
        raise ValueError("semantic policy requires semantic_scorer")

    sampling_started = time.perf_counter()
    ordered = tuple(sorted(frames, key=lambda frame: (frame.timestamp_s, frame.frame_id)))
    recent = recent_window(
        ordered,
        visible_until_s=query_time_s,
        capacity=recent_capacity,
        tolerance_s=tolerance_s,
    )
    recent_ids = {frame.frame_id for frame in recent}
    history = tuple(frame for frame in ordered if frame.frame_id not in recent_ids)

    retrieval_started = time.perf_counter()
    scores = semantic_scorer.score_frames(
        video_path=video_path,
        frames=history,
        images_by_id=images_by_id,
        query=query,
    )
    retrieval_finished = time.perf_counter()
    retrieval_latency_ms = (retrieval_finished - retrieval_started) * 1000.0
    selection = select_semantic_history(
        history,
        scores=scores,
        visible_until_s=query_time_s,
        capacity=memory_capacity,
        diversity=policy_kind == "semantic-diverse",
        candidate_multiplier=candidate_multiplier,
        cluster_window_s=cluster_window_s,
        diversity_weight=diversity_weight,
        tolerance_s=tolerance_s,
    )
    selected = tuple(
        sorted((*selection.frames, *recent), key=lambda frame: (frame.timestamp_s, frame.frame_id))
    )
    trace = tuple(
        {
            "frame_id": item.frame_id,
            "timestamp_s": item.timestamp_s,
            "semantic_score": item.semantic_score,
            "objective_score": item.objective_score,
        }
        for item in selection.trace
    )
    sampling_latency_ms = (
        (retrieval_started - sampling_started)
        + (time.perf_counter() - retrieval_finished)
    ) * 1000.0
    return selected, selection.frames, trace, retrieval_latency_ms, sampling_latency_ms


def _select_option_semantic_policy(
    frames: Sequence[FrameRecord],
    *,
    video_path: Path,
    images_by_id: Mapping[int, Any],
    query: str,
    options: Sequence[str],
    query_time_s: float,
    budget: int,
    policy_id: str,
    tolerance_s: float,
    semantic_scorer: OptionAwareFrameScorer | None,
    candidate_multiplier: int,
    cluster_window_s: float,
    diversity_weight: float,
) -> tuple[
    tuple[FrameRecord, ...],
    tuple[FrameRecord, ...],
    tuple[dict[str, Any], ...],
    float,
    float,
]:
    policy_kind, recent_capacity, memory_capacity = _policy_parts(policy_id, budget)
    score_option_frames = getattr(semantic_scorer, "score_option_frames", None)
    if not callable(score_option_frames):
        raise ValueError(
            "option-semantic policy requires a scorer with score_option_frames"
        )

    sampling_started = time.perf_counter()
    ordered = tuple(sorted(frames, key=lambda frame: (frame.timestamp_s, frame.frame_id)))
    recent = recent_window(
        ordered,
        visible_until_s=query_time_s,
        capacity=recent_capacity,
        tolerance_s=tolerance_s,
    )
    recent_ids = {frame.frame_id for frame in recent}
    history = tuple(frame for frame in ordered if frame.frame_id not in recent_ids)

    retrieval_started = time.perf_counter()
    scores_by_frame = score_option_frames(
        video_path=video_path,
        frames=history,
        images_by_id=images_by_id,
        question=query,
        options=options,
    )
    retrieval_finished = time.perf_counter()
    retrieval_latency_ms = (retrieval_finished - retrieval_started) * 1000.0
    option_scores: dict[int, OptionScoreItem] = aggregate_option_scores(
        scores_by_frame,
        expected_options=len(options),
    )
    selection = select_semantic_history(
        history,
        scores={frame_id: item.semantic_score for frame_id, item in option_scores.items()},
        visible_until_s=query_time_s,
        capacity=memory_capacity,
        diversity=policy_kind == "option-semantic-diverse",
        candidate_multiplier=candidate_multiplier,
        cluster_window_s=cluster_window_s,
        diversity_weight=diversity_weight,
        tolerance_s=tolerance_s,
    )
    selected = tuple(
        sorted((*selection.frames, *recent), key=lambda frame: (frame.timestamp_s, frame.frame_id))
    )
    trace = tuple(
        {
            "frame_id": item.frame_id,
            "timestamp_s": item.timestamp_s,
            "semantic_score": item.semantic_score,
            "objective_score": item.objective_score,
            "option_scores": list(option_scores[item.frame_id].option_scores),
            "best_option_index": option_scores[item.frame_id].best_option_index,
            "option_margin": option_scores[item.frame_id].option_margin,
        }
        for item in selection.trace
    )
    sampling_latency_ms = (
        (retrieval_started - sampling_started)
        + (time.perf_counter() - retrieval_finished)
    ) * 1000.0
    return selected, selection.frames, trace, retrieval_latency_ms, sampling_latency_ms


def _frame_manifest(frames: Sequence[FrameRecord]) -> list[dict[str, float | int]]:
    return [
        {"frame_id": frame.frame_id, "timestamp_s": frame.timestamp_s}
        for frame in frames
    ]


def evaluate_sample(
    sample: StreamingSample,
    *,
    config: ProjectConfig,
    frame_reader: Any,
    backend: ModelBackend,
    output_store: ResumeStore | None,
    policy_id: str = "recent",
    failure_store: FailureStore | None = None,
    config_id: str | None = None,
    record_metadata: Mapping[str, Any] | None = None,
    profile_store: RuntimeProfileStore | None = None,
    semantic_scorer: SemanticFrameScorer | OptionAwareFrameScorer | None = None,
) -> dict[str, Any]:
    """Evaluate one sample and release decoded images even on failure."""
    query_images: list[Any] = []

    class TrackingReader:
        def read_until(self, video_path: str | Path, end_time_s: float):
            for image, frame in frame_reader.read_until(video_path, end_time_s=end_time_s):
                query_images.append(image)
                yield image, frame

    try:
        return _evaluate_sample(
            sample,
            config=config,
            frame_reader=TrackingReader(),
            backend=backend,
            output_store=output_store,
            policy_id=policy_id,
            failure_store=failure_store,
            config_id=config_id,
            record_metadata=record_metadata,
            profile_store=profile_store,
            semantic_scorer=semantic_scorer,
        )
    finally:
        release = getattr(frame_reader, "release_query_images", None)
        if callable(release):
            release(sample.video_path, tuple(query_images))
        else:
            for image in query_images:
                close = getattr(image, "close", None)
                if close is not None:
                    close()
            gc.collect()
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


def _evaluate_sample(
    sample: StreamingSample,
    *,
    config: ProjectConfig,
    frame_reader: Any,
    backend: ModelBackend,
    output_store: ResumeStore | None,
    policy_id: str = "recent",
    failure_store: FailureStore | None = None,
    config_id: str | None = None,
    record_metadata: Mapping[str, Any] | None = None,
    profile_store: RuntimeProfileStore | None = None,
    semantic_scorer: SemanticFrameScorer | OptionAwareFrameScorer | None = None,
) -> dict[str, Any]:
    policy_kind, _, _ = _policy_parts(policy_id, config.experiment.visual_budget)
    total_started = time.perf_counter()
    query_time_s = sample.query_time_s
    tolerance_s = config.dataset.timestamp_tolerance_s
    context_start_s = max(0.0, query_time_s - config.dataset.context_horizon_s)

    decode_started = time.perf_counter()
    decoded = frame_reader.read_until(sample.video_path, end_time_s=query_time_s)
    buffer = CausalFrameBuffer(
        visible_until_s=query_time_s,
        context_start_s=context_start_s,
        timestamp_tolerance_s=tolerance_s,
    )
    images_by_id: dict[int, Any] = {}
    for image, frame in decoded:
        # This check occurs before context filtering so future data can never be hidden.
        buffer.add(frame)
        images_by_id[frame.frame_id] = image
    visible_frames = buffer.snapshot()
    decode_latency_ms = (time.perf_counter() - decode_started) * 1000.0

    memory_started = time.perf_counter()
    memory = BoundedTemporalMemory(
        capacity=config.experiment.visual_budget,
        visible_until_s=query_time_s,
        timestamp_tolerance_s=tolerance_s,
    )
    for frame in visible_frames:
        memory.update(frame)
    frozen_memory = memory.snapshot()
    memory_latency_ms = (time.perf_counter() - memory_started) * 1000.0

    if policy_kind in {"option-semantic", "option-semantic-diverse"}:
        (
            selected_frames,
            memory_frames,
            semantic_trace,
            retrieval_latency_ms,
            sampling_latency_ms,
        ) = _select_option_semantic_policy(
            visible_frames,
            video_path=sample.video_path,
            images_by_id=images_by_id,
            query=sample.question,
            options=sample.options,
            query_time_s=query_time_s,
            budget=config.experiment.visual_budget,
            policy_id=policy_id,
            tolerance_s=tolerance_s,
            semantic_scorer=semantic_scorer,
            candidate_multiplier=config.retrieval.candidate_multiplier,
            cluster_window_s=config.retrieval.cluster_window_s,
            diversity_weight=config.retrieval.diversity_weight,
        )
        include_timestamps = False
    elif policy_kind in {"semantic", "semantic-diverse"}:
        (
            selected_frames,
            memory_frames,
            semantic_trace,
            retrieval_latency_ms,
            sampling_latency_ms,
        ) = _select_semantic_policy(
            visible_frames,
            video_path=sample.video_path,
            images_by_id=images_by_id,
            query=sample.question,
            query_time_s=query_time_s,
            budget=config.experiment.visual_budget,
            policy_id=policy_id,
            tolerance_s=tolerance_s,
            semantic_scorer=semantic_scorer,
            candidate_multiplier=config.retrieval.candidate_multiplier,
            cluster_window_s=config.retrieval.cluster_window_s,
            diversity_weight=config.retrieval.diversity_weight,
        )
        include_timestamps = False
    else:
        sampling_started = time.perf_counter()
        selected_frames, memory_frames, include_timestamps = _select_policy(
            visible_frames,
            query_time_s=query_time_s,
            budget=config.experiment.visual_budget,
            policy_id=policy_id,
            tolerance_s=tolerance_s,
        )
        semantic_trace = ()
        retrieval_latency_ms = 0.0
        sampling_latency_ms = (time.perf_counter() - sampling_started) * 1000.0
    if selected_frames and max(frame.timestamp_s for frame in selected_frames) > query_time_s + tolerance_s:
        raise ValueError("selected frame is after the causal query boundary")
    selected_images = tuple(images_by_id[frame.frame_id] for frame in selected_frames)
    request = ModelRequest(
        images=selected_images,
        timestamps_s=tuple(frame.timestamp_s for frame in selected_frames),
        question=sample.question,
        options=sample.options,
        query_time_s=query_time_s,
        include_timestamps=include_timestamps,
    )
    # Construct once here as an auditable prompt contract, even when a backend builds it itself.
    prompt = build_prompt(
        question=sample.question,
        options=sample.options,
        timestamps_s=request.timestamps_s,
        query_time_s=query_time_s,
        include_timestamps=include_timestamps,
    )
    _reset_gpu_peak_memory()
    model_response = backend.infer(request)
    total_latency_ms = decode_latency_ms + model_response.latency_ms
    correct = model_response.answer == sample.answer
    retained_frames = len(selected_frames)
    record: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "video_path": str(sample.video_path),
        "config_id": config_id or policy_id,
        "policy_id": policy_id,
        "question": sample.question,
        "query_time_s": query_time_s,
        "ground_truth": sample.answer,
        "prediction": model_response.answer,
        "raw_prediction": model_response.raw_text,
        "correct": correct,
        "input_frames": len(selected_frames),
        "retained_frames": retained_frames,
        "source_frames": len(visible_frames),
        "compression_ratio": retained_frames / len(visible_frames) if visible_frames else 0.0,
        "selected_recent_frames": _frame_manifest(
            selected_frames if not memory_frames else tuple(
                frame for frame in selected_frames if frame not in memory_frames
            )
        ),
        "selected_memory_frames": _frame_manifest(memory_frames),
        "selected_frames": _frame_manifest(selected_frames),
        "timestamps_s": list(request.timestamps_s),
        "memory_snapshot": _frame_manifest(frozen_memory),
        "decode_latency_ms": decode_latency_ms,
        "model_latency_ms": model_response.latency_ms,
        "latency_ms": total_latency_ms,
        "peak_gpu_memory_mb": _peak_gpu_memory_mb(),
        "input_tokens": model_response.input_tokens,
        "output_tokens": model_response.output_tokens,
        "timestamp_prompt": prompt,
        "sample_fps": config.runtime.sample_fps,
        "retrieval_latency_ms": retrieval_latency_ms,
    }
    if policy_kind in {
        "semantic",
        "semantic-diverse",
        "option-semantic",
        "option-semantic-diverse",
    }:
        record["semantic_selection_trace"] = [dict(item) for item in semantic_trace]
    if record_metadata:
        record.update(dict(record_metadata))
    write_started = time.perf_counter()
    if output_store is not None:
        output_store.append(record)
    if failure_store is not None:
        failure_store.append(record)
    write_latency_ms = (time.perf_counter() - write_started) * 1000.0
    total_latency_ms = (time.perf_counter() - total_started) * 1000.0
    if profile_store is not None:
        profile_store.append(
            {
                "sample_id": sample.sample_id,
                "video_id": sample.video_id,
                "config_id": config_id or policy_id,
                "policy_id": policy_id,
                "decode_ms": decode_latency_ms,
                "sampling_ms": sampling_latency_ms,
                "retrieval_ms": retrieval_latency_ms,
                "memory_ms": memory_latency_ms,
                "processor_ms": model_response.processor_latency_ms,
                "generation_ms": model_response.generation_latency_ms,
                "postprocess_ms": model_response.postprocess_latency_ms,
                "write_ms": write_latency_ms,
                "total_ms": total_latency_ms,
            }
        )
    return record
