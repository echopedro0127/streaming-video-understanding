"""Option-aware semantic retrieval contracts and score aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Sequence


@dataclass(frozen=True)
class OptionScoreItem:
    """Aggregated option scores for one frame."""

    option_scores: tuple[float, ...]
    semantic_score: float
    best_option_index: int
    option_margin: float


def aggregate_option_scores(
    scores_by_frame: Mapping[int, Sequence[Real]],
    expected_options: int,
) -> dict[int, OptionScoreItem]:
    """Validate and aggregate per-option scores for each frame."""
    if type(expected_options) is not int:
        raise TypeError("expected_options must be an exact integer")
    if expected_options < 2:
        raise ValueError("expected_options must include at least 2 options")
    if not isinstance(scores_by_frame, Mapping):
        raise TypeError("scores_by_frame must be a mapping")

    aggregated: dict[int, OptionScoreItem] = {}
    for frame_id, scores in scores_by_frame.items():
        if type(frame_id) is not int:
            raise TypeError("frame IDs must be exact integer values")
        if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
            raise TypeError("option scores must be a sequence of real numeric values")
        if len(scores) != expected_options:
            raise ValueError(
                f"each frame must have exactly {expected_options} option scores"
            )

        normalized_scores: list[float] = []
        for score in scores:
            if isinstance(score, bool) or not isinstance(score, Real):
                raise TypeError("option scores must be real numeric values")
            normalized_score = float(score)
            if not math.isfinite(normalized_score):
                raise ValueError("option scores must be finite")
            normalized_scores.append(normalized_score)

        semantic_score = max(normalized_scores)
        best_option_index = normalized_scores.index(semantic_score)
        top_two = sorted(normalized_scores, reverse=True)[:2]
        aggregated[frame_id] = OptionScoreItem(
            option_scores=tuple(normalized_scores),
            semantic_score=semantic_score,
            best_option_index=best_option_index,
            option_margin=top_two[0] - top_two[1],
        )

    return aggregated
