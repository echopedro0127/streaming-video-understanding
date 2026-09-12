from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any


POLICY_ORDER = (
    "recent-8",
    "uniform-8",
    "recent-4+uniform-memory-4",
    "recent-4+timestamp-aware-memory-4",
)

POLICY_LABELS = {
    "recent-8": "Recent-8",
    "uniform-8": "Uniform-8",
    "recent-4+uniform-memory-4": "Recent-4 + Uniform-Memory-4",
    "recent-4+timestamp-aware-memory-4": "Recent-4 + Timestamp-aware-Memory-4",
}

_REQUIRED_FIELDS = (
    "question_accuracy",
    "p50_latency_ms",
    "p95_latency_ms",
    "peak_gpu_memory_mb",
    "average_input_frames",
    "average_retained_frames",
    "compression_ratio",
    "throughput_samples_per_s",
    "num_samples",
)


def _number(summary: Mapping[str, Any], field: str) -> float:
    if field not in summary:
        raise ValueError(f"summary is missing required field {field!r}")
    value = summary[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"summary field {field!r} must be numeric")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"summary field {field!r} must be finite")
    return value


def normalize_summary(
    summary: Mapping[str, Any], *, policy_id: str | None = None
) -> dict[str, Any]:
    """Validate a benchmark summary and normalize units for plotting."""
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a JSON object")
    resolved_policy = policy_id or summary.get("policy_id")
    if not isinstance(resolved_policy, str) or resolved_policy not in POLICY_LABELS:
        raise ValueError(f"unknown or missing policy_id: {resolved_policy!r}")
    values = {field: _number(summary, field) for field in _REQUIRED_FIELDS}
    if not 0.0 <= values["question_accuracy"] <= 1.0:
        raise ValueError("summary field 'question_accuracy' must be between 0 and 1")
    if values["num_samples"] <= 0 or values["num_samples"] != int(values["num_samples"]):
        raise ValueError("summary field 'num_samples' must be a positive integer")
    if values["p50_latency_ms"] < 0 or values["p95_latency_ms"] < 0:
        raise ValueError("summary latency fields must be non-negative")
    return {
        "policy_id": resolved_policy,
        "policy_label": POLICY_LABELS[resolved_policy],
        "question_accuracy": values["question_accuracy"],
        "p50_latency_s": values["p50_latency_ms"] / 1000.0,
        "p95_latency_s": values["p95_latency_ms"] / 1000.0,
        "peak_gpu_memory_mb": values["peak_gpu_memory_mb"],
        "average_input_frames": values["average_input_frames"],
        "retained_frames": values["average_retained_frames"],
        "compression_ratio": values["compression_ratio"],
        "throughput_samples_per_s": values["throughput_samples_per_s"],
        "num_samples": int(values["num_samples"]),
    }


def ordered_summary_rows(summaries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize summaries and return them in the fixed experiment order."""
    rows = [normalize_summary(summary) for summary in summaries]
    seen: set[str] = set()
    for row in rows:
        policy_id = str(row["policy_id"])
        if policy_id in seen:
            raise ValueError(f"duplicate summary for policy {policy_id!r}")
        seen.add(policy_id)
    order = {policy_id: index for index, policy_id in enumerate(POLICY_ORDER)}
    return sorted(rows, key=lambda row: order[str(row["policy_id"])])
