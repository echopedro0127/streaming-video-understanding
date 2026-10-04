import inspect
import math

import pytest

from streaming_video_understanding.retrieval import (
    OptionAwareFrameScorer,
    OptionScoreItem,
    aggregate_option_scores,
)


def test_aggregate_option_scores_uses_max_and_margin():
    result = aggregate_option_scores(
        {7: (0.1, 0.8, 0.4, 0.2)}, expected_options=4
    )

    assert result == {
        7: OptionScoreItem(
            option_scores=(0.1, 0.8, 0.4, 0.2),
            semantic_score=0.8,
            best_option_index=1,
            option_margin=0.4,
        )
    }


def test_aggregate_option_scores_allows_empty_mapping_after_validating_count():
    assert aggregate_option_scores({}, expected_options=2) == {}


def test_aggregate_option_scores_breaks_ties_by_lowest_option_index():
    result = aggregate_option_scores({5: (0.75, 0.75, 0.1)}, expected_options=3)

    assert result[5].best_option_index == 0
    assert result[5].semantic_score == pytest.approx(0.75)
    assert result[5].option_margin == pytest.approx(0.0)


def test_aggregate_option_scores_rejects_wrong_option_count():
    with pytest.raises(ValueError, match="exactly 3"):
        aggregate_option_scores({7: (0.1, 0.8)}, expected_options=3)


@pytest.mark.parametrize("expected_options", [0, 1])
def test_aggregate_option_scores_rejects_fewer_than_two_options(expected_options):
    with pytest.raises(ValueError, match="at least 2"):
        aggregate_option_scores({}, expected_options=expected_options)


@pytest.mark.parametrize("score", [math.inf, -math.inf, math.nan])
def test_aggregate_option_scores_rejects_non_finite_scores(score):
    with pytest.raises(ValueError, match="finite"):
        aggregate_option_scores({7: (score, 0.2)}, expected_options=2)


def test_aggregate_option_scores_rejects_non_numeric_scores():
    with pytest.raises(TypeError, match="real numeric"):
        aggregate_option_scores({7: ("high", 0.2)}, expected_options=2)


@pytest.mark.parametrize("frame_id", [True, 7.0, "7"])
def test_aggregate_option_scores_rejects_non_exact_integer_frame_ids(frame_id):
    with pytest.raises(TypeError, match="exact integer"):
        aggregate_option_scores({frame_id: (0.1, 0.2)}, expected_options=2)


@pytest.mark.parametrize("expected_options", [True, 2.0, "2"])
def test_aggregate_option_scores_rejects_non_exact_integer_option_counts(
    expected_options,
):
    with pytest.raises(TypeError, match="exact integer"):
        aggregate_option_scores({}, expected_options=expected_options)


def test_option_aware_frame_scorer_protocol_has_question_and_options_only():
    parameters = inspect.signature(
        OptionAwareFrameScorer.score_option_frames
    ).parameters

    assert list(parameters) == [
        "self",
        "video_path",
        "frames",
        "images_by_id",
        "question",
        "options",
    ]
    assert "answer" not in parameters
