from __future__ import annotations

import json

import pytest

from streaming_video_understanding.analysis.pairs import (
    analyze_pair_records,
    representative_rows,
    write_pair_artifacts,
)


RECENT_POLICY = "recent-8"
OPTION_POLICY = "recent-4+option-semantic-memory-4"


def _timestamp_prompt(question: str, options: tuple[str, ...]) -> str:
    lines = [f"Question: {question}", "Options:"]
    lines.extend(f"{letter}. {option}" for letter, option in zip("ABCD", options))
    return "\n".join(lines)


def _record(
    sample_number: int,
    policy_id: str,
    prediction: str,
    *,
    correct: bool | None = None,
    question: str = "What happened before the person stopped?",
    ground_truth: str = "A",
    query_time_s: float = 10.0,
    options: tuple[str, ...] | None = None,
    future_frame: bool = False,
) -> dict[str, object]:
    option_text = options or ("opened the door", "picked up a cup", "sat down", "left")
    recent_frames = [
        {"frame_id": 8, "timestamp_s": 8.0},
        {"frame_id": 9, "timestamp_s": 9.0},
    ]
    memory_frames = [{"frame_id": 3, "timestamp_s": 3.0}]
    if future_frame:
        recent_frames[-1] = {"frame_id": 99, "timestamp_s": query_time_s + 0.1}
    record: dict[str, object] = {
        "sample_id": f"question-{sample_number}",
        "video_id": f"camera_sample_{sample_number}",
        "policy_id": policy_id,
        "question": question,
        "query_time_s": query_time_s,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "selected_recent_frames": recent_frames,
        "selected_memory_frames": memory_frames if policy_id == OPTION_POLICY else [],
        "selected_frames": recent_frames + (memory_frames if policy_id == OPTION_POLICY else []),
        "timestamps_s": [frame["timestamp_s"] for frame in recent_frames]
        + ([3.0] if policy_id == OPTION_POLICY else []),
        "timestamp_prompt": _timestamp_prompt(question, option_text),
    }
    if correct is not None:
        record["correct"] = correct
    if options is not None:
        record["options"] = list(options)
    if policy_id == OPTION_POLICY:
        record["semantic_selection_trace"] = [
            {
                "frame_id": 3,
                "timestamp_s": 3.0,
                "semantic_score": 0.82,
                "objective_score": 0.74,
                "option_scores": [0.80, 0.20, 0.10, 0.05],
                "best_option_index": 0,
                "option_margin": 0.60,
            }
        ]
        record["memory_snapshot"] = [
            {"frame_id": 3, "timestamp_s": 3.0, "score": 0.82}
        ]
    return record


def _paired_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    outcomes = (("A", "A"), ("B", "A"), ("A", "B"), ("C", "B"))
    recent = [_record(index, RECENT_POLICY, recent_prediction) for index, (recent_prediction, _) in enumerate(outcomes, 1)]
    option = [_record(index, OPTION_POLICY, option_prediction) for index, (_, option_prediction) in enumerate(outcomes, 1)]
    return recent, option


def test_analyze_pair_records_classifies_all_four_outcomes_and_exposes_diagnostics() -> None:
    recent, option = _paired_records()

    result = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)

    assert [row["group"] for row in result.rows] == [
        "both_correct",
        "memory_helped",
        "memory_hurt",
        "both_wrong",
    ]
    assert result.summary["n_pairs"] == 4
    assert result.summary["group_counts"] == {
        "both_correct": 1,
        "memory_helped": 1,
        "memory_hurt": 1,
        "both_wrong": 1,
    }
    assert result.summary["group_percentages"] == {
        "both_correct": 25.0,
        "memory_helped": 25.0,
        "memory_hurt": 25.0,
        "both_wrong": 25.0,
    }
    assert result.summary["recent_accuracy"] == pytest.approx(0.5)
    assert result.summary["option_accuracy"] == pytest.approx(0.5)
    assert result.summary["net_gain"] == 0
    assert result.summary["accuracy_delta"] == pytest.approx(0.0)
    assert result.summary["accuracy_delta_check"] is True

    row = result.rows[1]
    assert row["recent_timestamps"] == [8.0, 9.0]
    assert row["recent_recent_timestamps"] == row["recent_timestamps"]
    assert row["option_recent_timestamps"] == [8.0, 9.0]
    assert row["option_memory_timestamps"] == [3.0]
    assert row["option_semantic_trace"][0]["option_scores"] == [0.8, 0.2, 0.1, 0.05]
    assert row["per_frame_option_scores"] == [{"frame_id": 3, "scores": [0.8, 0.2, 0.1, 0.05]}]
    assert row["option_margin"] == pytest.approx(0.6)
    assert row["retrieval_top_option"] == "A"
    assert "h1_recent_frame_loss_proxy" in row
    assert "h2_retrieval_top_option_match" in row
    assert "temporal_keyword_present" in row


def test_pair_analysis_is_order_independent_and_sorted_by_sample_id() -> None:
    recent, option = _paired_records()

    first = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)
    second = analyze_pair_records(
        list(reversed(recent)), list(reversed(option)), RECENT_POLICY, OPTION_POLICY
    )

    assert first.rows == second.rows
    assert first.summary == second.summary
    assert [row["sample_id"] for row in first.rows] == [
        "question-1",
        "question-2",
        "question-3",
        "question-4",
    ]


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda recent, option: recent.extend([dict(recent[0])]), "duplicate"),
        (lambda recent, option: option.pop(), "missing paired"),
    ],
)
def test_pair_analysis_rejects_duplicate_or_missing_pairs(mutator, match: str) -> None:
    recent, option = _paired_records()
    mutator(recent, option)

    with pytest.raises(ValueError, match=match):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


@pytest.mark.parametrize("field", ["question", "ground_truth", "query_time_s"])
def test_pair_analysis_rejects_metadata_mismatch(field: str) -> None:
    recent, option = _paired_records()
    option[0][field] = {
        "question": "A different question",
        "ground_truth": "D",
        "query_time_s": 10.01,
    }[field]

    with pytest.raises(ValueError, match=field):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


def test_pair_analysis_rejects_mismatched_options_and_future_frames() -> None:
    recent, option = _paired_records()
    option[0]["options"] = ["different", "options", "than", "these"]

    with pytest.raises(ValueError, match="options"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)

    recent, option = _paired_records()
    recent[0] = _record(1, RECENT_POLICY, "A", future_frame=True)

    with pytest.raises(ValueError, match="future frame"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


def test_representative_rows_limit_and_stable_sorting() -> None:
    recent, option = _paired_records()
    result = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)

    helped = representative_rows(list(reversed(result.rows)), "memory_helped", limit=20)
    assert [row["sample_id"] for row in helped] == ["question-2"]
    assert representative_rows(result.rows, "both_wrong", limit=0) == []
    assert representative_rows(result.rows, "memory_helped", limit=1) == helped


def test_write_pair_artifacts_serializes_json_csv_and_group_jsonl(tmp_path) -> None:
    recent, option = _paired_records()
    result = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)

    paths = write_pair_artifacts(result, tmp_path / "paired", representative_limit=1)

    assert set(paths) == {"csv", "summary_json", "helped_jsonl", "hurt_jsonl"}
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8"))["n_pairs"] == 4
    assert paths["helped_jsonl"].read_text(encoding="utf-8").count("\n") == 1
    assert paths["hurt_jsonl"].read_text(encoding="utf-8").count("\n") == 1
    header = paths["csv"].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "recent_recent_timestamps" in header
    assert "option_semantic_trace" in header
