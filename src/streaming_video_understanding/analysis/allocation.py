"""Allocation diagnostics on validated QA pairs; no inference or parameter tuning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import random
from typing import Any

from ..streaming.routing import route_question


def _validated_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("at least one validated pair is required")
    seen = set()
    result = []
    for row in rows:
        for field in ("sample_id", "video_id", "question"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"paired row requires non-empty {field}")
        if row["sample_id"] in seen:
            raise ValueError(f"duplicate sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        for field in ("recent_correct", "option_correct"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"{field} must be boolean")
        left, right = row["recent_correct"], row["option_correct"]
        expected = ("both_correct" if left and right else "memory_hurt" if left
                    else "memory_helped" if right else "both_wrong")
        if row.get("group") != expected:
            raise ValueError("paired group disagrees with correctness")
        result.append(dict(row))
    return sorted(result, key=lambda row: (row["video_id"], row["sample_id"]))


def _quantile(values: list[float], q: float) -> float:
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    helped = sum(row["group"] == "memory_helped" for row in rows)
    hurt = sum(row["group"] == "memory_hurt" for row in rows)
    return {
        "n_pairs": len(rows), "helped": helped, "hurt": hurt,
        "baseline_accuracy": sum(row["recent_correct"] for row in rows) / len(rows),
        "candidate_accuracy": sum(row["option_correct"] for row in rows) / len(rows),
        "accuracy_delta": (helped - hurt) / len(rows),
    }


def allocation_diagnosis(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 42, n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """QA-micro delta CI by resampling videos and keeping their QA clusters intact."""
    if type(n_bootstrap) is not int or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    ordered = _validated_rows(rows)
    videos = defaultdict(list)
    groups = defaultdict(list)
    for row in ordered:
        videos[row["video_id"]].append(row)
        groups[route_question(row["question"]).group].append(row)
    clusters = [
        (sum(int(row["option_correct"]) - int(row["recent_correct"]) for row in cluster), len(cluster))
        for _, cluster in sorted(videos.items())
    ]
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_bootstrap):
        sampled = rng.choices(clusters, k=len(clusters))
        estimates.append(sum(delta for delta, _ in sampled) / sum(count for _, count in sampled))
    estimates.sort()
    return {
        **_counts(ordered), "n_videos": len(clusters),
        "video_macro_accuracy_delta": sum(delta / count for delta, count in clusters) / len(clusters),
        "bootstrap": {"unit": "video", "method": "paired cluster percentile",
                      "seed": seed, "n_resamples": n_bootstrap,
                      "ci95": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
                      "warning": "one video cannot estimate between-video uncertainty" if len(clusters) < 2 else None},
        "route_proxy_groups": {key: _counts(value) for key, value in sorted(groups.items())},
        "grouping_note": "Question-rules-v1 groups are lexical proxies, not evidence labels or calibrated confidence.",
    }


def evidence_review_queue(
    rows: Sequence[Mapping[str, Any]], *, limit: int = 60, seed: int = 42,
) -> list[dict[str, Any]]:
    """Group-balanced review sample, blinded to correctness and policy selection.

    This is diagnostic case sampling, not an unbiased accuracy estimate. Review
    the video up to query_time_s and annotate sufficient evidence intervals first.
    Empty intervals with pending status mean unannotated, never no evidence.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    ordered = _validated_rows(rows)
    groups = defaultdict(list)
    for row in ordered:
        groups[row["group"]].append(row)
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    selected = []
    while len(selected) < min(limit, len(ordered)):
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                row = groups[key].pop()
                selected.append({
                    "sample_id": row["sample_id"], "video_id": row["video_id"],
                    "question": row["question"], "options": row.get("options"),
                    "query_time_s": row["query_time_s"],
                    "annotation_status": "pending", "evidence_type": None,
                    "evidence_available": None, "evidence_intervals": [],
                    "annotator": "", "notes": "",
                })
    # Keep outcome balancing, but remove the positional outcome pattern.
    rng.shuffle(selected)
    return selected
