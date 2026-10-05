from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from streaming_video_understanding.analysis.pairs import (
    PairAnalysisResult,
    analyze_pair_records,
    representative_rows,
    write_pair_artifacts,
)


RECENT_POLICY = "recent-8"
OPTION_POLICY = "recent-4+option-semantic-memory-4"


def _load_pair_analysis_cli():
    script_path = Path(__file__).parents[1] / "scripts" / "analyze_policy_pairs.py"
    spec = importlib.util.spec_from_file_location("analyze_policy_pairs_cli", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load CLI module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert row["recent_frame_ids"] == [8, 9]
    assert row["recent_recent_frame_ids"] == row["recent_frame_ids"]
    assert row["option_recent_frame_ids"] == [8, 9]
    assert row["option_memory_frame_ids"] == [3]
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


def test_pair_analysis_filters_exact_policies_and_inclusive_video_index_range() -> None:
    recent, option = _paired_records()
    recent.append(_record(99, f"{RECENT_POLICY}-suffix", "A"))
    option.append(_record(99, f"{OPTION_POLICY}-suffix", "A"))

    result = analyze_pair_records(
        recent,
        option,
        RECENT_POLICY,
        OPTION_POLICY,
        min_video_index=2,
        max_video_index=3,
    )

    assert [row["sample_id"] for row in result.rows] == ["question-2", "question-3"]
    assert result.summary["n_pairs"] == 2


def test_h4_rate_uses_option_wrong_denominator_and_exposes_counts() -> None:
    recent, option = _paired_records()

    result = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)

    assert result.summary["both_wrong_count"] == 1
    assert result.summary["option_wrong_count"] == 2
    assert result.summary["h4_both_wrong_count"] == 1
    assert result.summary["h4_option_wrong_count"] == 2
    assert result.summary["h4_both_wrong_option_wrong_count"] == 1
    assert result.summary["h4_both_wrong_option_wrong_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("record_kind", "field", "missing_key"),
    [
        ("recent", "selected_frames", "frame_id"),
        ("recent", "selected_recent_frames", "timestamp_s"),
        ("option", "selected_memory_frames", "frame_id"),
        ("option", "memory_snapshot", "timestamp_s"),
        ("option", "semantic_selection_trace", "timestamp_s"),
    ],
)
def test_pair_analysis_rejects_frame_mappings_missing_required_keys(
    record_kind: str, field: str, missing_key: str
) -> None:
    recent, option = _paired_records()
    record = recent[0] if record_kind == "recent" else option[0]
    frames = record[field]
    assert isinstance(frames, list)
    malformed = dict(frames[0])
    del malformed[missing_key]
    frames[0] = malformed

    with pytest.raises(ValueError, match=field):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


@pytest.mark.parametrize(
    ("record_kind", "field"),
    [
        ("recent", "timestamps_s"),
        ("option", "memory_snapshot"),
        ("option", "semantic_selection_trace"),
    ],
)
def test_pair_analysis_rejects_future_trace_and_timestamp_values(
    record_kind: str, field: str
) -> None:
    recent, option = _paired_records()
    record = recent[0] if record_kind == "recent" else option[0]
    future_timestamp = float(record["query_time_s"]) + 0.1
    if field == "timestamps_s":
        record[field] = [future_timestamp]
    else:
        frames = record[field]
        assert isinstance(frames, list)
        frames[0] = dict(frames[0])
        frames[0]["timestamp_s"] = future_timestamp

    with pytest.raises(ValueError, match="future frame"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


def test_h1_recent_frame_loss_uses_ids_when_timestamps_overlap() -> None:
    recent, option = _paired_records()
    baseline_recent = [
        {"frame_id": 100, "timestamp_s": 8.0},
        {"frame_id": 101, "timestamp_s": 9.0},
    ]
    option_recent = [
        {"frame_id": 200, "timestamp_s": 8.0},
        {"frame_id": 201, "timestamp_s": 9.0},
    ]
    recent[0]["selected_recent_frames"] = baseline_recent
    recent[0]["selected_frames"] = baseline_recent
    recent[0]["timestamps_s"] = [8.0, 9.0]
    option[0]["selected_recent_frames"] = option_recent
    option[0]["selected_frames"] = option_recent + option[0]["selected_memory_frames"]
    option[0]["timestamps_s"] = [8.0, 9.0, 3.0]

    row = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY).rows[0]

    assert row["recent_frame_ids"] == [100, 101]
    assert row["option_recent_frame_ids"] == [200, 201]
    assert row["recent_timestamps"] == row["option_recent_timestamps"]
    assert row["h1_recent_frame_loss_proxy"] is True
    assert row["h1_recent_frame_loss_count"] == 2


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
    assert paths["csv"].name == "paired-paired.csv"
    assert paths["summary_json"].name == "paired-summary.json"
    assert paths["helped_jsonl"].name == "paired-helped.jsonl"
    assert paths["hurt_jsonl"].name == "paired-hurt.jsonl"
    header = paths["csv"].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {
        "options",
        "recent_frame_ids",
        "option_memory_frame_ids",
        "recent_frame_temporal_distances",
        "option_memory_frame_temporal_distances",
        "memory_semantic_scores",
        "option_margins",
        "option_semantic_trace",
    } <= set(header)


def test_memory_semantic_scores_merge_trace_scores_by_frame_id() -> None:
    recent, option = _paired_records()
    option[0]["memory_snapshot"] = [{"frame_id": 3, "timestamp_s": 3.0}]

    row = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY).rows[0]

    assert row["memory_semantic_scores"] == row["memory_scores"]
    assert row["memory_semantic_scores"][0]["frame_id"] == 3
    assert row["memory_semantic_scores"][0]["semantic_score"] == pytest.approx(0.82)
    assert row["memory_semantic_scores"][0]["objective_score"] == pytest.approx(0.74)
    assert set(row["memory_semantic_scores"][0]) == {
        "frame_id",
        "timestamp_s",
        "score",
        "semantic_score",
        "objective_score",
    }


def test_pair_artifacts_reject_unexpected_csv_fields(tmp_path) -> None:
    recent, option = _paired_records()
    result = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)
    malformed = PairAnalysisResult(
        rows=[{**result.rows[0], "unexpected_field": "must not be discarded"}],
        summary=result.summary,
    )

    with pytest.raises(ValueError, match="not in fieldnames"):
        write_pair_artifacts(malformed, tmp_path / "paired")


@pytest.mark.parametrize(
    ("container", "field"),
    [
        ("trace", "semantic_score"),
        ("trace", "objective_score"),
        ("snapshot", "score"),
    ],
)
def test_pair_analysis_rejects_non_finite_score_values(
    container: str, field: str
) -> None:
    recent, option = _paired_records()
    if container == "trace":
        trace = option[0]["semantic_selection_trace"]
        assert isinstance(trace, list)
        trace[0][field] = float("nan")
    else:
        snapshot = option[0]["memory_snapshot"]
        assert isinstance(snapshot, list)
        snapshot[0][field] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


def test_pair_artifacts_reject_non_finite_json_values(tmp_path) -> None:
    recent, option = _paired_records()
    result = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)
    malformed = PairAnalysisResult(
        rows=result.rows,
        summary={**result.summary, "non_finite": float("nan")},
    )

    with pytest.raises(ValueError, match="out of range|finite"):
        write_pair_artifacts(malformed, tmp_path / "paired")


def test_pair_analysis_rejects_numeric_frame_scalars_and_non_mapping_trace_entries() -> None:
    recent, option = _paired_records()
    recent[0]["selected_recent_frames"] = [8.0]

    with pytest.raises(ValueError, match="frame object"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)

    recent, option = _paired_records()
    option[0]["semantic_selection_trace"] = [3.0]

    with pytest.raises(ValueError, match="frame object"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


@pytest.mark.parametrize(
    ("record_kind", "field"),
    [
        ("recent", "selected_recent_frames"),
        ("option", "memory_snapshot"),
        ("option", "semantic_selection_trace"),
    ],
)
def test_pair_analysis_rejects_duplicate_frame_ids_in_manifests(
    record_kind: str, field: str
) -> None:
    recent, option = _paired_records()
    record = recent[0] if record_kind == "recent" else option[0]
    frames = record[field]
    assert isinstance(frames, list)
    frames.append(dict(frames[0]))

    with pytest.raises(ValueError, match="duplicate"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


@pytest.mark.parametrize(
    "bad_options",
    [
        ["first", "second", "third"],
        ["first", "second", "third", "fourth", "fifth"],
        {"A": "first", "B": "second", "D": "fourth"},
        {"A": "first", "B": "second", "C": "third", "D": "fourth", "E": "fifth"},
    ],
)
def test_pair_analysis_rejects_non_exact_explicit_options(bad_options) -> None:
    recent, option = _paired_records()
    recent[0]["options"] = bad_options
    option[0]["options"] = bad_options

    with pytest.raises(ValueError, match="options"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


@pytest.mark.parametrize(
    "prompt",
    [
        "Question: q\nOptions:\nA. one\nA. duplicate\nB. two\nC. three\nD. four",
        "Question: q\nOptions:\nA. one\nB. two\nD. four",
    ],
)
def test_pair_analysis_rejects_duplicate_or_sparse_prompt_options(prompt: str) -> None:
    recent, option = _paired_records()
    recent[0].pop("options", None)
    option[0].pop("options", None)
    recent[0]["timestamp_prompt"] = prompt
    option[0]["timestamp_prompt"] = prompt

    with pytest.raises(ValueError, match="options"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


def test_pair_analysis_requires_boolean_correct_field() -> None:
    recent, option = _paired_records()
    recent[0]["correct"] = "false"

    with pytest.raises(ValueError, match="correct"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


def test_pair_analysis_h2_excludes_unavailable_rows_and_h3_exposes_counts() -> None:
    recent, option = _paired_records()
    option[0].pop("semantic_selection_trace")
    option[0].pop("memory_snapshot")

    summary = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY).summary

    assert summary["h2_retrieval_top_option_match_rate"] == pytest.approx(1 / 3)
    assert summary["h2_retrieval_top_option_match_count"] == 1
    assert summary["h2_retrieval_top_option_match_evaluated_count"] == 3
    assert summary["h2_retrieval_top_option_match_count_by_group"] == {
        "both_correct": 0,
        "memory_helped": 1,
        "memory_hurt": 0,
        "both_wrong": 0,
    }
    assert summary["h3_temporal_keyword_counts_by_group"] == {
        "both_correct": 1,
        "memory_helped": 1,
        "memory_hurt": 1,
        "both_wrong": 1,
    }


def test_pair_rows_and_csv_include_canonical_timestamp_distance_fields(tmp_path) -> None:
    recent, option = _paired_records()

    result = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)
    row = result.rows[0]

    assert row["recent_timestamp_distances_s"] == [2.0, 1.0]
    assert row["option_memory_timestamp_distances_s"] == [7.0]
    assert row["recent_timestamp_distances_s"] == row["recent_frame_distances_s"]
    assert row["option_memory_timestamp_distances_s"] == row["option_memory_frame_distances_s"]

    paths = write_pair_artifacts(result, tmp_path / "paired")
    header = paths["csv"].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "recent_timestamp_distances_s" in header
    assert "option_memory_timestamp_distances_s" in header


@pytest.mark.parametrize("frame_id", [None, "8", 8.0, True])
def test_pair_analysis_rejects_non_exact_integer_frame_ids(frame_id) -> None:
    recent, option = _paired_records()
    frames = recent[0]["selected_recent_frames"]
    assert isinstance(frames, list)
    frames[0]["frame_id"] = frame_id

    with pytest.raises(ValueError, match="frame_id"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


@pytest.mark.parametrize(
    "bad_options",
    [
        ["first", "", "third", "fourth"],
        ["first", "   ", "third", "fourth"],
        ["first", 2, "third", "fourth"],
    ],
)
def test_pair_analysis_rejects_empty_or_non_string_options(bad_options) -> None:
    recent, option = _paired_records()
    recent[0]["options"] = bad_options
    option[0]["options"] = bad_options

    with pytest.raises(ValueError, match="options"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


@pytest.mark.parametrize(
    "prompt",
    [
        "Question: q\nOptions:\nA) one\nB. two\nC. three\nD. four",
        "Question: q\nOptions:\nA. \nB. two\nC. three\nD. four",
        "Question: q\nOptions:\nA. one\nB. two\nC. three\nD. four\nE. extra",
    ],
)
def test_pair_analysis_requires_canonical_non_empty_prompt_options(prompt: str) -> None:
    recent, option = _paired_records()
    recent[0].pop("options", None)
    option[0].pop("options", None)
    recent[0]["timestamp_prompt"] = prompt
    option[0]["timestamp_prompt"] = prompt

    with pytest.raises(ValueError, match="options"):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("option_scores", [0.8, 0.2, 0.1]),
        ("option_scores", [0.8, float("nan"), 0.1, 0.05]),
        ("best_option_index", True),
        ("best_option_index", 4),
        ("best_option_index", -1),
        ("best_option_index", 1.0),
        ("option_margin", "not numeric"),
        ("option_margin", float("inf")),
    ],
)
def test_pair_analysis_rejects_invalid_option_trace_fields(field: str, value) -> None:
    recent, option = _paired_records()
    trace = option[0]["semantic_selection_trace"]
    assert isinstance(trace, list)
    trace[0][field] = value

    with pytest.raises(ValueError, match=field):
        analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY)


def test_pair_analysis_option_margins_include_derived_scores_and_match_aggregate() -> None:
    recent, option = _paired_records()
    trace = option[0]["semantic_selection_trace"]
    assert isinstance(trace, list)
    trace[0].pop("option_margin")

    row = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY).rows[0]

    assert row["option_margins"] == [pytest.approx(0.6)]
    assert row["option_margin"] == max(row["option_margins"])


def test_pair_analysis_accuracy_delta_uses_correct_counts_directly() -> None:
    recent = [_record(index, RECENT_POLICY, "A") for index in range(1, 4)]
    option_predictions = ("B", "A", "B")
    option = [
        _record(index, OPTION_POLICY, prediction)
        for index, prediction in enumerate(option_predictions, 1)
    ]

    summary = analyze_pair_records(recent, option, RECENT_POLICY, OPTION_POLICY).summary

    assert summary["accuracy_delta"] == (1 - 3) / 3


def test_pair_analysis_cli_main_writes_expected_artifacts_from_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recent, option = _paired_records()
    recent_input = tmp_path / "inputs" / "recent.jsonl"
    option_input = tmp_path / "inputs" / "option.jsonl"
    recent_input.parent.mkdir()
    recent_input.write_text(
        "".join(json.dumps(record) + "\n" for record in recent), encoding="utf-8"
    )
    option_input.write_text(
        "".join(json.dumps(record) + "\n" for record in option), encoding="utf-8"
    )
    output_dir = tmp_path / "new" / "analysis"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_policy_pairs.py",
            "--recent-input",
            str(recent_input),
            "--recent-policy",
            RECENT_POLICY,
            "--option-input",
            str(option_input),
            "--option-policy",
            OPTION_POLICY,
            "--output-dir",
            str(output_dir),
            "--output-prefix",
            "contract",
            "--video-min",
            "1",
            "--video-max",
            "4",
            "--representative-limit",
            "1",
        ],
    )

    cli = _load_pair_analysis_cli()
    assert cli.main() == 0

    captured = capsys.readouterr()
    summary_path = output_dir / "contract-summary.json"
    assert "paired=4" in captured.out
    assert str(summary_path) in captured.out
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "contract-helped.jsonl",
        "contract-hurt.jsonl",
        "contract-paired.csv",
        "contract-summary.json",
    ]
    assert json.loads(summary_path.read_text(encoding="utf-8"))["n_pairs"] == 4
