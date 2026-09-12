from __future__ import annotations

import pytest

from streaming_video_understanding.analysis.summaries import (
    normalize_summary,
    ordered_summary_rows,
)


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
