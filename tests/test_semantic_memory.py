import math
from dataclasses import FrozenInstanceError

import pytest

from streaming_video_understanding.data.schema import FrameRecord
from streaming_video_understanding.streaming.buffer import FutureFrameLeakageError
from streaming_video_understanding.streaming.semantic import (
    SemanticSelection,
    SemanticSelectionItem,
    select_semantic_history,
)


def _frames(count: int = 8) -> tuple[FrameRecord, ...]:
    return tuple(FrameRecord(frame_id, float(frame_id)) for frame_id in range(count))


def test_selects_highest_semantic_scores_and_returns_chronological_trace():
    result = select_semantic_history(
        _frames(),
        scores={frame_id: float(frame_id) for frame_id in range(8)},
        visible_until_s=7.0,
        capacity=3,
    )

    assert isinstance(result, SemanticSelection)
    assert [frame.frame_id for frame in result.frames] == [5, 6, 7]
    assert [item.frame_id for item in result.trace] == [5, 6, 7]
    assert [item.timestamp_s for item in result.trace] == [5.0, 6.0, 7.0]
    assert [item.semantic_score for item in result.trace] == [5.0, 6.0, 7.0]
    assert [item.objective_score for item in result.trace] == [5.0, 6.0, 7.0]


def test_diversity_avoids_selecting_only_a_high_scoring_cluster():
    scores = {0: 0.98, 1: 0.10, 2: 0.99, 3: 0.98, 4: 0.97, 5: 0.10, 6: 0.10, 7: 0.98}

    result = select_semantic_history(
        _frames(),
        scores=scores,
        visible_until_s=7.0,
        capacity=3,
        diversity=True,
        candidate_multiplier=3,
        cluster_window_s=2.0,
        diversity_weight=0.5,
    )

    selected_ids = [frame.frame_id for frame in result.frames]
    assert 2 in selected_ids
    assert selected_ids != [2, 3, 4]
    assert selected_ids == [0, 2, 7]
    assert [item.frame_id for item in result.trace] == selected_ids
    assert result.trace[0].objective_score == pytest.approx(
        (0.98 - 0.10) / (0.99 - 0.10) - 0.5 * math.exp(-1.0)
    )
    assert result.trace[1].objective_score == pytest.approx(1.0)
    assert result.trace[2].objective_score == pytest.approx(
        (0.98 - 0.10) / (0.99 - 0.10) - 0.5 * math.exp(-2.5)
    )


def test_equal_score_tie_returns_earliest_frames_without_diversity():
    result = select_semantic_history(
        _frames(4),
        scores={frame_id: 1.0 for frame_id in range(4)},
        visible_until_s=3.0,
        capacity=2,
    )

    assert [frame.frame_id for frame in result.frames] == [0, 1]


def test_equal_diversity_scores_normalize_relevance_to_one():
    frames = (FrameRecord(2, 10.0), FrameRecord(1, 1.0), FrameRecord(0, 0.0))
    result = select_semantic_history(
        frames,
        scores={0: 5.0, 1: 5.0, 2: 5.0},
        visible_until_s=10.0,
        capacity=2,
        diversity=True,
        candidate_multiplier=2,
    )

    assert [frame.frame_id for frame in result.frames] == [0, 2]
    assert [item.objective_score for item in result.trace] == pytest.approx(
        [1.0, 1.0 - 0.25 * math.exp(-10.0 / 3.0)]
    )


def test_missing_score_raises_clear_error():
    with pytest.raises(ValueError, match="missing semantic score"):
        select_semantic_history(
            _frames(2), scores={0: 0.5}, visible_until_s=1.0, capacity=1
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "0.5", True])
def test_score_must_be_finite_and_numeric(score):
    with pytest.raises(ValueError, match="semantic score"):
        select_semantic_history(
            _frames(1), scores={0: score}, visible_until_s=0.0, capacity=1
        )


def test_future_frame_raises_instead_of_being_filtered():
    with pytest.raises(FutureFrameLeakageError):
        select_semantic_history(
            _frames(3), scores={0: 0.0, 1: 1.0, 2: 2.0}, visible_until_s=1.0, capacity=2
        )


def test_capacity_zero_still_performs_causal_validation_before_returning_empty():
    with pytest.raises(FutureFrameLeakageError):
        select_semantic_history(
            _frames(2), scores={}, visible_until_s=0.0, capacity=0
        )

    result = select_semantic_history(
        _frames(1), scores={}, visible_until_s=0.0, capacity=0
    )
    assert result == SemanticSelection(frames=(), trace=())


@pytest.mark.parametrize("timestamp", [-1.0, float("nan"), float("inf")])
def test_rejects_invalid_frame_timestamps(timestamp):
    with pytest.raises(ValueError, match="timestamp"):
        select_semantic_history(
            (FrameRecord(0, timestamp),), scores={0: 1.0}, visible_until_s=1.0, capacity=1
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"capacity": -1}, "capacity"),
        ({"candidate_multiplier": 0}, "candidate_multiplier"),
        ({"candidate_multiplier": -1}, "candidate_multiplier"),
        ({"cluster_window_s": 0.0}, "cluster_window_s"),
        ({"cluster_window_s": float("inf")}, "cluster_window_s"),
        ({"diversity_weight": -0.01}, "diversity_weight"),
        ({"diversity_weight": 1.01}, "diversity_weight"),
        ({"diversity_weight": float("nan")}, "diversity_weight"),
        ({"tolerance_s": -1.0}, "tolerance_s"),
        ({"tolerance_s": float("inf")}, "tolerance_s"),
    ],
)
def test_validates_selector_parameters(overrides, message):
    arguments = {
        "scores": {},
        "visible_until_s": 0.0,
        "capacity": 0,
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        select_semantic_history((), **arguments)


def test_empty_input_returns_empty_selection():
    assert select_semantic_history(
        (), scores={}, visible_until_s=0.0, capacity=2
    ) == SemanticSelection(frames=(), trace=())


def test_public_result_types_are_immutable():
    item = SemanticSelectionItem(0, 0.0, 1.0, 1.0)
    selection = SemanticSelection((FrameRecord(0, 0.0),), (item,))

    with pytest.raises(FrozenInstanceError):
        item.objective_score = 0.0
    with pytest.raises(FrozenInstanceError):
        selection.frames = ()
