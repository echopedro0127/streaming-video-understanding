from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

from ..data.schema import FrameRecord
from .buffer import FutureFrameLeakageError


@dataclass(frozen=True)
class SemanticSelectionItem:
    frame_id: int
    timestamp_s: float
    semantic_score: float
    objective_score: float


@dataclass(frozen=True)
class SemanticSelection:
    frames: tuple[FrameRecord, ...]
    trace: tuple[SemanticSelectionItem, ...]


def _is_finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _validate_parameters(
    capacity: int,
    candidate_multiplier: int,
    cluster_window_s: float,
    diversity_weight: float,
    tolerance_s: float,
) -> None:
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")
    if (
        not isinstance(candidate_multiplier, int)
        or isinstance(candidate_multiplier, bool)
        or candidate_multiplier <= 0
    ):
        raise ValueError("candidate_multiplier must be a positive integer")
    if not _is_finite_number(cluster_window_s) or cluster_window_s <= 0:
        raise ValueError("cluster_window_s must be finite and positive")
    if (
        not _is_finite_number(diversity_weight)
        or diversity_weight < 0
        or diversity_weight > 1
    ):
        raise ValueError("diversity_weight must be finite and in [0, 1]")
    if not _is_finite_number(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and non-negative")


def _causal_frames(
    frames: Sequence[FrameRecord],
    visible_until_s: float,
    tolerance_s: float,
) -> tuple[FrameRecord, ...]:
    for frame in frames:
        if not _is_finite_number(frame.timestamp_s) or frame.timestamp_s < 0:
            raise ValueError(f"invalid frame timestamp: {frame.timestamp_s!r}")

    ordered = tuple(sorted(frames, key=lambda frame: (frame.timestamp_s, frame.frame_id)))
    for frame in ordered:
        if frame.timestamp_s > visible_until_s + tolerance_s:
            raise FutureFrameLeakageError(
                f"Frame {frame.frame_id} at {frame.timestamp_s:.6f}s is after "
                f"visible time {visible_until_s:.6f}s"
            )
    return ordered


def _semantic_scores(
    frames: Sequence[FrameRecord], scores: Mapping[int, float]
) -> dict[int, float]:
    validated: dict[int, float] = {}
    for frame in frames:
        if frame.frame_id not in scores:
            raise ValueError(f"missing semantic score for frame {frame.frame_id}")
        score = scores[frame.frame_id]
        if not _is_finite_number(score):
            raise ValueError(
                f"semantic score for frame {frame.frame_id} must be finite and numeric"
            )
        validated[frame.frame_id] = float(score)
    return validated


def _trace_item(
    frame: FrameRecord, semantic_score: float, objective_score: float
) -> SemanticSelectionItem:
    return SemanticSelectionItem(
        frame_id=frame.frame_id,
        timestamp_s=frame.timestamp_s,
        semantic_score=semantic_score,
        objective_score=objective_score,
    )


def select_semantic_history(
    frames: Sequence[FrameRecord],
    *,
    scores: Mapping[int, float],
    visible_until_s: float,
    capacity: int,
    diversity: bool = False,
    candidate_multiplier: int = 4,
    cluster_window_s: float = 3.0,
    diversity_weight: float = 0.25,
    tolerance_s: float = 1e-6,
) -> SemanticSelection:
    """Select causal history frames by relevance, optionally penalizing redundancy."""
    _validate_parameters(
        capacity,
        candidate_multiplier,
        cluster_window_s,
        diversity_weight,
        tolerance_s,
    )
    ordered = _causal_frames(frames, visible_until_s, tolerance_s)
    if capacity == 0 or not ordered:
        return SemanticSelection(frames=(), trace=())

    semantic_scores = _semantic_scores(ordered, scores)
    ranked = sorted(
        ordered,
        key=lambda frame: (
            -semantic_scores[frame.frame_id],
            frame.timestamp_s,
            frame.frame_id,
        ),
    )

    if not diversity:
        selected = sorted(
            ranked[:capacity], key=lambda frame: (frame.timestamp_s, frame.frame_id)
        )
        return SemanticSelection(
            frames=tuple(selected),
            trace=tuple(
                _trace_item(
                    frame,
                    semantic_scores[frame.frame_id],
                    semantic_scores[frame.frame_id],
                )
                for frame in selected
            ),
        )

    pool = ranked[: min(len(ranked), capacity * candidate_multiplier)]
    pool_scores = [semantic_scores[frame.frame_id] for frame in pool]
    minimum_score = min(pool_scores)
    maximum_score = max(pool_scores)
    if minimum_score == maximum_score:
        relevance = {frame.frame_id: 1.0 for frame in pool}
    else:
        score_range = maximum_score - minimum_score
        relevance = {
            frame.frame_id: (semantic_scores[frame.frame_id] - minimum_score) / score_range
            for frame in pool
        }

    remaining = list(pool)
    chosen: list[tuple[FrameRecord, float]] = []
    while remaining and len(chosen) < capacity:
        evaluated: list[tuple[FrameRecord, float]] = []
        for candidate in remaining:
            redundancy = max(
                (
                    math.exp(
                        -abs(candidate.timestamp_s - selected.timestamp_s)
                        / cluster_window_s
                    )
                    for selected, _ in chosen
                ),
                default=0.0,
            )
            objective = relevance[candidate.frame_id] - diversity_weight * redundancy
            evaluated.append((candidate, objective))

        selected, objective = min(
            evaluated,
            key=lambda item: (
                -item[1],
                -relevance[item[0].frame_id],
                item[0].timestamp_s,
                item[0].frame_id,
            ),
        )
        chosen.append((selected, objective))
        remaining.remove(selected)

    chronological = sorted(chosen, key=lambda item: (item[0].timestamp_s, item[0].frame_id))
    return SemanticSelection(
        frames=tuple(frame for frame, _ in chronological),
        trace=tuple(
            _trace_item(frame, semantic_scores[frame.frame_id], objective)
            for frame, objective in chronological
        ),
    )
