from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _finite_values(records: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and isfinite(float(value)):
            values.append(float(value))
    return values


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = len(rows)
    correct = sum(bool(row.get("correct", False)) for row in rows)
    latencies = _finite_values(rows, "latency_ms")

    by_video: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row.get("video_id") is not None:
            by_video[str(row["video_id"])].append(bool(row.get("correct", False)))
    video_scores = [sum(values) / len(values) for values in by_video.values() if values]

    input_frames = _finite_values(rows, "input_frames")
    retained_frames = _finite_values(rows, "retained_frames")
    compression_values = _finite_values(rows, "compression_ratio")
    peak_memory = _finite_values(rows, "peak_gpu_memory_mb")
    total_latency_s = sum(latencies) / 1000.0
    labels = ("A", "B", "C", "D")
    ground_truth_counts = Counter(
        str(row["ground_truth"]) for row in rows if row.get("ground_truth") is not None
    )
    prediction_counts = Counter(
        str(row["prediction"]) for row in rows if row.get("prediction") is not None
    )
    position_accuracy = {
        label: (
            sum(
                bool(row.get("correct", False))
                for row in rows
                if row.get("ground_truth") == label
            )
            / ground_truth_counts[label]
            if ground_truth_counts[label]
            else 0.0
        )
        for label in labels
    }

    return {
        "num_samples": total,
        "question_accuracy": correct / total if total else 0.0,
        "video_macro_accuracy": sum(video_scores) / len(video_scores) if video_scores else 0.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 50.0),
        "p95_latency_ms": _percentile(latencies, 95.0),
        "throughput_samples_per_s": total / total_latency_s if total_latency_s > 0 else 0.0,
        "average_input_frames": sum(input_frames) / len(input_frames) if input_frames else 0.0,
        "average_retained_frames": sum(retained_frames) / len(retained_frames)
        if retained_frames
        else 0.0,
        "compression_ratio": sum(compression_values) / len(compression_values)
        if compression_values
        else 0.0,
        "peak_gpu_memory_mb": max(peak_memory, default=0.0),
        "ground_truth_distribution": {
            label: int(ground_truth_counts[label]) for label in labels
        },
        "prediction_distribution": {
            label: int(prediction_counts[label]) for label in labels
        },
        "accuracy_by_ground_truth_position": position_accuracy,
    }
