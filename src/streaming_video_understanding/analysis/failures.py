from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .summaries import POLICY_ORDER


REVIEW_FIELDS = (
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
)

REVIEW_CATEGORIES = (
    "forgotten_early_evidence",
    "temporal_ordering_error",
    "distractor_frames",
    "insufficient_recent_context",
    "memory_compression_loss",
    "semantic_retrieval_failure",
    "timestamp_confusion",
    "other",
)


def _review_id(record: Mapping[str, Any]) -> str:
    return f"{record['policy_id']}::{record['sample_id']}"


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"failure record field {field!r} must be numeric")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"failure record field {field!r} must be finite")
    return result


def validate_causal_manifest(record: Mapping[str, Any], tolerance_s: float = 1e-6) -> None:
    """Reject failure evidence that contains a frame after its query time."""
    query_time_s = _finite_number(record.get("query_time_s"), "query_time_s")
    selected_frames = record.get("selected_frames")
    if not isinstance(selected_frames, list):
        raise ValueError("failure record is missing selected_frames evidence")
    for frame in selected_frames:
        if not isinstance(frame, Mapping):
            raise ValueError("selected_frames must contain JSON objects")
        timestamp_s = _finite_number(frame.get("timestamp_s"), "timestamp_s")
        if timestamp_s > query_time_s + tolerance_s:
            raise ValueError(
                f"future frame timestamp {timestamp_s} exceeds query time {query_time_s}"
            )


def _reviewable(record: Mapping[str, Any]) -> bool:
    if record.get("selected_frames") is None and record.get("status") in {
        "failed",
        "error",
    }:
        return False
    return True


def select_failures(
    records: Iterable[Mapping[str, Any]], *, limit: int = 30, seed: int = 7
) -> list[dict[str, Any]]:
    """Select stable, policy-balanced, deduplicated failure evidence."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("failure records must be JSON objects")
        if not _reviewable(record):
            continue
        sample_id = record.get("sample_id")
        policy_id = record.get("policy_id")
        if not isinstance(sample_id, str) or not isinstance(policy_id, str):
            raise ValueError("failure record requires string sample_id and policy_id")
        validate_causal_manifest(record)
        key = (sample_id, policy_id)
        candidate = dict(record)
        candidate["review_id"] = _review_id(candidate)
        candidate["category"] = ""
        candidate["notes"] = ""
        unique.setdefault(key, candidate)

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(
        unique.values(), key=lambda item: (str(item["policy_id"]), str(item["sample_id"]))
    ):
        groups.setdefault(str(record["policy_id"]), []).append(record)
    policy_order = {policy_id: index for index, policy_id in enumerate(POLICY_ORDER)}
    policies = sorted(groups, key=lambda policy_id: (policy_order.get(policy_id, len(POLICY_ORDER)), policy_id))
    rng = random.Random(seed)
    for policy_id in policies:
        rng.shuffle(groups[policy_id])

    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        added = False
        for policy_id in policies:
            if groups[policy_id] and len(selected) < limit:
                selected.append(groups[policy_id].pop())
                added = True
        if not added:
            break
    return selected


def write_failure_review(records: Iterable[Mapping[str, Any]], output_dir: Path) -> dict[str, Path]:
    """Write selected evidence and blank human-review fields."""
    rows = [dict(record) for record in records]
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "selected_failures.jsonl"
    csv_path = output_dir / "failure_review.csv"
    markdown_path = output_dir / "failure_review.md"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REVIEW_FIELDS})
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Failure Review\n\n")
        handle.write(f"Selected {len(rows)} deterministic failure records.\n\n")
        handle.write("Assign one category and add evidence-based notes manually.\n\n")
        handle.write("Allowed categories: " + ", ".join(f"`{item}`" for item in REVIEW_CATEGORIES) + ".\n\n")
        handle.write("| review_id | policy_id | sample_id | query_time_s | prediction | ground_truth | category | notes |\n")
        handle.write("|---|---|---|---:|---|---|---|---|\n")
        for row in rows:
            values = []
            for field in ("review_id", "policy_id", "sample_id", "query_time_s", "prediction", "ground_truth", "category", "notes"):
                value = str(row.get(field, "")).replace("|", "\\|").replace("\n", " ")
                values.append(value)
            handle.write("| " + " | ".join(values) + " |\n")
    return {"jsonl": jsonl_path, "csv": csv_path, "markdown": markdown_path}
