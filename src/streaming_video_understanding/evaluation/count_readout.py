"""Paired, shared-image count readout and a question-only pilot sampler."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
from typing import Any

from ..analysis.allocation import allocation_diagnosis
from ..models.base import ModelRequest, ModelResponse
from ..models.counting import numeric_options
from ..streaming.routing import route_question
from .metrics import _percentile


class PairedCountBackend:
    """Complete both generations before the evaluator releases decoded images."""

    def __init__(self, backend, *, count_first: bool = False):
        self.backend = backend
        self.count_first = count_first
        self.responses: dict[str, ModelResponse] = {}

    def infer(self, request: ModelRequest) -> ModelResponse:
        if request.readout_mode != "mcq" or request.include_timestamps:
            raise ValueError("paired count v1 requires an untimestamped MCQ request")
        numeric_options(request.options)
        self.responses = {}
        for mode in (("count", "mcq") if self.count_first else ("mcq", "count")):
            # replace preserves the exact image tuple and timestamps; no labels here.
            item = replace(request, readout_mode=mode, allow_invalid_answer=(mode == "mcq"))
            self.responses[mode] = self.backend.infer(item)
        return self.responses["mcq"]


def select_counting_cases(samples, *, excluded_video_ids: set[str], seed: str) -> tuple[str, ...]:
    videos = defaultdict(list)
    for sample in samples:
        if sample.video_id in excluded_video_ids or route_question(sample.question).group != "counting":
            continue
        try:
            numeric_options(sample.options)
        except ValueError:
            continue
        videos[sample.video_id].append(sample)
    selected = []
    for video_id, rows in sorted(videos.items()):
        selected.append(min(rows, key=lambda row: hashlib.sha256(
            f"{seed}:{video_id}:{row.sample_id}".encode()).hexdigest()).sample_id)
    return tuple(sorted(selected))


def summarize_count_pairs(rows: list[dict[str, Any]], *, n_bootstrap: int = 2000) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty count pilot")
    paired = []
    methods = {}
    for mode in ("mcq", "count"):
        valid = [row for row in rows if row["methods"][mode]["count"] is not None]
        correct = sum(row["methods"][mode]["count"] == row["ground_truth_count"] for row in rows)
        methods[mode] = {
            "correct": correct, "accuracy": correct / len(rows),
            "parse_valid_count": len(valid), "parse_valid_rate": len(valid) / len(rows),
            "outside_option_count": sum(row["methods"][mode]["count"] not in row["option_counts"] for row in valid),
            "valid_count_mae": sum(abs(row["methods"][mode]["count"] - row["ground_truth_count"])
                                   for row in valid) / len(valid) if valid else None,
            "valid_count_mae_denominator": len(valid),
        }
        latencies = [row["methods"][mode]["model_latency_ms"] for row in rows
                     if "model_latency_ms" in row["methods"][mode]]
        if latencies:
            methods[mode]["mean_model_latency_ms"] = sum(latencies) / len(latencies)
            methods[mode]["p50_model_latency_ms"] = _percentile(latencies, 50.0)
            methods[mode]["p95_model_latency_ms"] = _percentile(latencies, 95.0)
    for row in rows:
        left = row["methods"]["mcq"]["count"] == row["ground_truth_count"]
        right = row["methods"]["count"]["count"] == row["ground_truth_count"]
        paired.append(dict(sample_id=row["sample_id"], video_id=row["video_id"],
                           question=row.get("question", "Counting question"),
                           recent_correct=left, option_correct=right,
                           group="both_correct" if left and right else "memory_hurt" if left
                           else "memory_helped" if right else "both_wrong"))
    diagnosis = allocation_diagnosis(paired, n_bootstrap=n_bootstrap)
    return {"num_samples": len(rows), "num_videos": len({row["video_id"] for row in rows}),
            "methods": methods,
            "paired": {"accuracy_delta": diagnosis["accuracy_delta"],
                       "helped": diagnosis["helped"], "hurt": diagnosis["hurt"],
                       "bootstrap": diagnosis["bootstrap"]},
            "scope": "Counting pilot, video-disjoint from the 11-case diagnostic, not pristine unseen data"}
