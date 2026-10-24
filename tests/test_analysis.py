from __future__ import annotations

import json

import pytest

from streaming_video_understanding.analysis.failures import (
    select_failures,
    validate_causal_manifest,
    write_failure_review,
)
from streaming_video_understanding.analysis.summaries import (
    ALLOCATION_ABLATION_POLICY_ORDER,
    DISPLAY_POLICY_ORDER,
    OPTION_SEMANTIC_POLICY_ORDER,
    POLICY_ORDER,
    POLICY_LABELS,
    build_comparison_rows,
    normalize_summary,
    ordered_summary_rows,
)
from scripts.plot_results import _annotation_offsets


def _summary(policy_id: str = "recent-8") -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "question_accuracy": 0.652,
        "p50_latency_ms": 33705.9,
        "p95_latency_ms": 98533.8,
        "peak_gpu_memory_mb": 3537.3,
        "average_input_frames": 7.944,
        "average_retained_frames": 7.944,
        "compression_ratio": 0.1912,
        "throughput_samples_per_s": 0.0032,
        "num_samples": 250,
    }


@pytest.mark.parametrize("policy_id", ["adaptive-semantic-v1", "recent-6+semantic-memory-2"])
def test_selective_memory_policies_are_plot_compatible(policy_id):
    row = normalize_summary(_summary(policy_id))
    assert row["policy_id"] == policy_id
    assert policy_id in DISPLAY_POLICY_ORDER


def test_normalize_summary_maps_metrics_and_converts_latency_to_seconds() -> None:
    row = normalize_summary(_summary())

    assert row["policy_id"] == "recent-8"
    assert row["policy_label"] == "Recent-8"
    assert row["question_accuracy"] == pytest.approx(0.652)
    assert row["p50_latency_s"] == pytest.approx(33.7059)
    assert row["p95_latency_s"] == pytest.approx(98.5338)
    assert row["peak_gpu_memory_mb"] == pytest.approx(3537.3)
    assert row["average_input_frames"] == pytest.approx(7.944)
    assert row["retained_frames"] == pytest.approx(7.944)
    assert row["compression_ratio"] == pytest.approx(0.1912)
    assert row["throughput_samples_per_s"] == pytest.approx(0.0032)
    assert row["num_samples"] == 250


def test_normalize_summary_rejects_missing_required_metric() -> None:
    summary = _summary()
    del summary["question_accuracy"]

    with pytest.raises(ValueError, match="question_accuracy"):
        normalize_summary(summary)


def test_ordered_summary_rows_use_fixed_policy_order() -> None:
    summaries = [_summary("uniform-8"), _summary("recent-8")]
    summaries[0]["question_accuracy"] = 0.628

    rows = ordered_summary_rows(summaries)

    assert [row["policy_id"] for row in rows] == ["recent-8", "uniform-8"]


def test_policy_labels_include_semantic_memory_methods() -> None:
    assert POLICY_LABELS["recent-4+semantic-memory-4"] == "Recent-4 + Semantic-Memory-4"
    assert POLICY_LABELS["recent-4+semantic-diverse-memory-4"] == (
        "Recent-4 + Semantic-Diverse-Memory-4"
    )


def test_option_semantic_policy_order_and_labels() -> None:
    assert OPTION_SEMANTIC_POLICY_ORDER == (
        "recent-4+option-semantic-memory-4",
        "recent-4+option-semantic-diverse-memory-4",
    )
    assert POLICY_LABELS["recent-4+option-semantic-memory-4"] == (
        "Recent-4 + Option-Semantic-Memory-4"
    )
    assert POLICY_LABELS["recent-4+option-semantic-diverse-memory-4"] == (
        "Recent-4 + Option-Semantic-Diverse-Memory-4"
    )


def test_6plus2_policy_has_display_label_but_not_fixed_baseline_order() -> None:
    policy_id = "recent-6+option-semantic-memory-2"

    assert POLICY_LABELS[policy_id] == "Recent-6 + Option-Semantic-Memory-2"
    assert policy_id not in POLICY_ORDER
    assert DISPLAY_POLICY_ORDER[-1] == policy_id


def test_display_policy_order_appends_option_semantic_methods() -> None:
    assert DISPLAY_POLICY_ORDER[-3:-1] == OPTION_SEMANTIC_POLICY_ORDER
    assert DISPLAY_POLICY_ORDER[-1:] == ALLOCATION_ABLATION_POLICY_ORDER


def test_ordered_summary_rows_put_semantic_methods_after_fixed_methods() -> None:
    policy_ids = [
        "recent-8",
        "uniform-8",
        "recent-4+uniform-memory-4",
        "recent-4+timestamp-aware-memory-4",
        "recent-4+semantic-memory-4",
        "recent-4+semantic-diverse-memory-4",
        "recent-4+option-semantic-memory-4",
        "recent-4+option-semantic-diverse-memory-4",
    ]

    rows = ordered_summary_rows([_summary(policy_id) for policy_id in reversed(policy_ids)])

    assert [row["policy_id"] for row in rows] == policy_ids


def _failure(sample_id: str, policy_id: str, *, frame_time: float = 4.0) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "policy_id": policy_id,
        "video_id": sample_id.split("_")[0],
        "question": "Which event happened?",
        "query_time_s": 4.0,
        "prediction": "A",
        "ground_truth": "B",
        "selected_frames": [{"frame_id": 0, "timestamp_s": frame_time}],
        "selected_recent_frames": [],
        "selected_memory_frames": [],
    }


def test_validate_causal_manifest_rejects_future_frame() -> None:
    with pytest.raises(ValueError, match="future frame"):
        validate_causal_manifest(_failure("sample_1", "recent-8", frame_time=4.1))


def test_select_failures_is_stable_balanced_and_deduplicated() -> None:
    records = [
        _failure("sample_2", "uniform-8"),
        _failure("sample_1", "recent-8"),
        _failure("sample_1", "recent-8"),
        _failure("sample_3", "recent-8"),
        _failure("sample_4", "recent-4+uniform-memory-4"),
    ]

    first = select_failures(records, limit=3, seed=7)
    second = select_failures(list(reversed(records)), limit=3, seed=7)

    assert [row["review_id"] for row in first] == [row["review_id"] for row in second]
    assert len({(row["sample_id"], row["policy_id"]) for row in first}) == 3
    assert {row["policy_id"] for row in first} == {
        "recent-8",
        "uniform-8",
        "recent-4+uniform-memory-4",
    }


def test_write_failure_review_writes_expected_csv_headers(tmp_path) -> None:
    records = select_failures(
        [_failure("sample_1", "recent-8"), _failure("sample_2", "uniform-8")],
        limit=2,
        seed=7,
    )

    paths = write_failure_review(records, tmp_path)

    assert set(paths) == {"jsonl", "csv", "markdown"}
    header = paths["csv"].read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == [
        "review_id",
        "sample_id",
        "policy_id",
        "video_id",
        "question",
        "query_time_s",
        "prediction",
        "ground_truth",
        "category",
        "notes",
    ]


def test_build_comparison_rows_uses_fixed_policy_order_and_seconds(tmp_path) -> None:
    policy_ids = [
        "recent-8",
        "uniform-8",
        "recent-4+uniform-memory-4",
        "recent-4+timestamp-aware-memory-4",
    ]
    paths = []
    for policy_id in reversed(policy_ids):
        path = tmp_path / f"full-{policy_id}-summary.json"
        path.write_text(json.dumps(_summary(policy_id)), encoding="utf-8")
        paths.append(path)

    rows = build_comparison_rows(paths)

    assert [row["policy_id"] for row in rows] == policy_ids
    assert rows[0]["p50_latency_s"] == pytest.approx(33.7059)
    assert all(row["num_samples"] == 250 for row in rows)


def test_build_comparison_rows_rejects_wrong_sample_count(tmp_path) -> None:
    path = tmp_path / "full-recent-8-summary.json"
    summary = _summary()
    summary["num_samples"] = 249
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="250"):
        build_comparison_rows([path])


def test_build_comparison_rows_accepts_hyphenated_real_result_filenames(tmp_path) -> None:
    filenames = [
        "full-recent-8-summary.json",
        "full-uniform-8-summary.json",
        "full-recent-4-uniform-memory-4-summary.json",
        "full-recent-4-timestamp-aware-memory-4-summary.json",
    ]
    paths = []
    for filename in filenames:
        path = tmp_path / filename
        summary = _summary()
        summary.pop("policy_id")
        summary["question_accuracy"] = 0.64
        path.write_text(json.dumps(summary), encoding="utf-8")
        paths.append(path)

    rows = build_comparison_rows(paths)

    assert [row["policy_id"] for row in rows] == [
        "recent-8",
        "uniform-8",
        "recent-4+uniform-memory-4",
        "recent-4+timestamp-aware-memory-4",
    ]


def test_annotation_offsets_separate_collocated_labels() -> None:
    offsets = _annotation_offsets(4)

    assert len(set(offsets)) == 4
