from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from math import isfinite
from pathlib import Path
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


def _policy_from_summary_path(path: Path) -> str:
    prefix = "full-"
    suffix = "-summary.json"
    slug = path.name[len(prefix) : -len(suffix)] if path.name.startswith(prefix) and path.name.endswith(suffix) else ""
    for policy_id in POLICY_ORDER:
        if slug in {policy_id, policy_id.replace("+", "-")}:
            return policy_id
    raise ValueError(
        f"cannot infer fixed policy from summary filename {path.name!r}; "
        "expected full-<policy>-summary.json"
    )


def build_comparison_rows(
    summary_paths: Iterable[str | Path], *, expected_samples: int = 250
) -> list[dict[str, Any]]:
    """Load and validate one summary for every fixed-budget policy."""
    if isinstance(expected_samples, bool) or expected_samples <= 0:
        raise ValueError("expected_samples must be a positive integer")
    summaries: list[dict[str, Any]] = []
    for raw_path in summary_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"summary file does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed summary JSON: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"summary must be a JSON object: {path}")
        inferred_policy_id = _policy_from_summary_path(path)
        payload_policy_id = payload.get("policy_id")
        if payload_policy_id is not None and str(payload_policy_id) != inferred_policy_id:
            raise ValueError(
                f"summary policy_id {payload_policy_id!r} does not match filename policy "
                f"{inferred_policy_id!r}: {path}"
            )
        policy_id = inferred_policy_id
        row = normalize_summary(payload, policy_id=str(policy_id))
        if row["num_samples"] != expected_samples:
            raise ValueError(
                f"summary {path} has {row['num_samples']} samples; expected {expected_samples}"
            )
        summaries.append({**payload, "policy_id": row["policy_id"]})
    rows = ordered_summary_rows(summaries)
    if {row["policy_id"] for row in rows} != set(POLICY_ORDER):
        missing = sorted(set(POLICY_ORDER) - {row["policy_id"] for row in rows})
        extra = sorted({row["policy_id"] for row in rows} - set(POLICY_ORDER))
        raise ValueError(f"summary set must contain all fixed policies; missing={missing}, extra={extra}")
    return rows
