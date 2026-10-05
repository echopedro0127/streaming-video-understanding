"""Paired diagnosis helpers for comparing two streaming policies."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
import re
from typing import Any


_TIMESTAMP_TOLERANCE_S = 1e-6
_VIDEO_INDEX_RE = re.compile(r"(?:^|_)sample_(\d+)$")
_OPTION_LINE_RE = re.compile(r"^\s*([A-D])\s*[.):\]-]\s*(.*?)\s*$")
_TEMPORAL_KEYWORDS = (
    "before",
    "after",
    "first",
    "then",
    "currently",
    "now",
    "still",
    "previously",
    "earlier",
    "later",
    "continue",
    "stop",
    "change",
)
_GROUPS = ("both_correct", "memory_helped", "memory_hurt", "both_wrong")


@dataclass(frozen=True)
class PairAnalysisResult:
    """JSON-serializable paired rows and aggregate diagnosis."""

    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _jsonable(value: Any) -> Any:
    """Convert ordinary record values to deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _video_index(video_id: Any) -> int:
    if not isinstance(video_id, str):
        raise ValueError("video_id must be a string")
    match = _VIDEO_INDEX_RE.search(video_id)
    if match is None:
        raise ValueError(
            f"video_id {video_id!r} does not end with sample_<index>"
        )
    return int(match.group(1))


def _validate_range(value: int | None, field: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{field} must be an integer or None")


def _filtered_records(
    records: Iterable[Mapping[str, Any]],
    policy_id: str,
    *,
    min_video_index: int | None,
    max_video_index: int | None,
    label: str,
) -> dict[str, dict[str, Any]]:
    if isinstance(records, (str, bytes)):
        raise ValueError(f"{label}_records must be an iterable of mappings")
    selected: dict[str, dict[str, Any]] = {}
    try:
        iterator = iter(records)
    except TypeError as exc:
        raise ValueError(f"{label}_records must be an iterable of mappings") from exc

    for raw_record in iterator:
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"{label} records must be JSON objects")
        if raw_record.get("policy_id") != policy_id:
            continue
        sample_id = raw_record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} record requires sample_id")
        index = _video_index(raw_record.get("video_id"))
        if min_video_index is not None and index < min_video_index:
            continue
        if max_video_index is not None and index > max_video_index:
            continue
        if sample_id in selected:
            raise ValueError(f"duplicate {label} record for sample_id {sample_id!r}")
        candidate = dict(raw_record)
        candidate["_video_index"] = index
        selected[sample_id] = candidate
    return selected


def _prompt_options(prompt: Any) -> tuple[str, ...] | None:
    if not isinstance(prompt, str):
        return None
    parsed: dict[str, str] = {}
    for line in prompt.splitlines():
        match = _OPTION_LINE_RE.match(line)
        if match is not None:
            parsed[match.group(1)] = match.group(2).strip()
    if not parsed:
        return None
    letters = "ABCD"
    if not all(letter in parsed for letter in letters if letter <= max(parsed)):
        return None
    return tuple(parsed[letter] for letter in letters if letter in parsed)


def _record_options(record: Mapping[str, Any]) -> tuple[str, ...] | None:
    if "options" in record and record.get("options") is not None:
        options = record.get("options")
        if isinstance(options, Mapping):
            values = []
            for letter in "ABCD":
                if letter not in options:
                    break
                values.append(str(options[letter]))
            if values:
                return tuple(values)
            raise ValueError("options mapping must use A-D keys")
        if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
            raise ValueError("options must be a sequence")
        return tuple(str(option) for option in options)
    return _prompt_options(record.get("timestamp_prompt"))


def _frame_items(value: Any, field: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a list of frame objects")
    if not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a list of frame objects")
    items: list[dict[str, Any]] = []
    for index, raw_frame in enumerate(value):
        if isinstance(raw_frame, Mapping):
            if "timestamp_s" not in raw_frame:
                continue
            timestamp_s = _finite_number(raw_frame["timestamp_s"], f"{field}.timestamp_s")
            frame_id = raw_frame.get("frame_id", index)
            item = dict(raw_frame)
            item["frame_id"] = frame_id
            item["timestamp_s"] = timestamp_s
        else:
            timestamp_s = _finite_number(raw_frame, f"{field}.timestamp_s")
            item = {"frame_id": index, "timestamp_s": timestamp_s}
        items.append(item)
    return sorted(items, key=lambda item: (float(item["timestamp_s"]), str(item["frame_id"])))


def _validate_causal_timestamps(record: Mapping[str, Any], query_time_s: float) -> None:
    fields = (
        "selected_frames",
        "selected_recent_frames",
        "selected_memory_frames",
        "memory_snapshot",
        "semantic_selection_trace",
        "option_semantic_trace",
    )
    for field in fields:
        for frame in _frame_items(record.get(field), field):
            timestamp_s = float(frame["timestamp_s"])
            if timestamp_s > query_time_s + _TIMESTAMP_TOLERANCE_S:
                raise ValueError(
                    f"future frame timestamp {timestamp_s} exceeds query time {query_time_s}"
                )
    timestamps_s = record.get("timestamps_s")
    if timestamps_s is not None:
        if isinstance(timestamps_s, (str, bytes)) or not isinstance(timestamps_s, Sequence):
            raise ValueError("timestamps_s must be a sequence")
        for timestamp in timestamps_s:
            timestamp_s = _finite_number(timestamp, "timestamps_s")
            if timestamp_s > query_time_s + _TIMESTAMP_TOLERANCE_S:
                raise ValueError(
                    f"future frame timestamp {timestamp_s} exceeds query time {query_time_s}"
                )


def _record_frames(record: Mapping[str, Any], *, memory: bool) -> list[dict[str, Any]]:
    field = "selected_memory_frames" if memory else "selected_recent_frames"
    frames = _frame_items(record.get(field), field)
    if frames:
        return frames
    selected = _frame_items(record.get("selected_frames"), "selected_frames")
    if memory:
        recent_ids = {
            frame.get("frame_id")
            for frame in _frame_items(record.get("selected_recent_frames"), "selected_recent_frames")
        }
        return [frame for frame in selected if frame.get("frame_id") not in recent_ids]
    memory_ids = {
        frame.get("frame_id") for frame in _frame_items(record.get("selected_memory_frames"), "selected_memory_frames")
    }
    return [frame for frame in selected if frame.get("frame_id") not in memory_ids]


def _timestamps(frames: Sequence[Mapping[str, Any]]) -> list[float]:
    return [float(frame["timestamp_s"]) for frame in frames]


def _temporal_distances(timestamps: Sequence[float], query_time_s: float) -> list[float]:
    return [query_time_s - timestamp for timestamp in timestamps]


def _traces(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_trace = record.get("semantic_selection_trace")
    if raw_trace is None:
        raw_trace = record.get("option_semantic_trace")
    if raw_trace is None:
        return []
    if isinstance(raw_trace, (str, bytes)) or not isinstance(raw_trace, Sequence):
        raise ValueError("option semantic trace must be a list")
    trace = [dict(item) for item in raw_trace if isinstance(item, Mapping)]
    return sorted(
        trace,
        key=lambda item: (
            float(item.get("timestamp_s", 0.0)),
            str(item.get("frame_id", "")),
        ),
    )


def _option_diagnostics(trace: Sequence[Mapping[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], float | None, str | None
]:
    per_frame: list[dict[str, Any]] = []
    margins: list[float] = []
    for index, item in enumerate(trace):
        raw_scores = item.get("option_scores")
        scores: list[float] | None = None
        if isinstance(raw_scores, Sequence) and not isinstance(raw_scores, (str, bytes)):
            scores = [_finite_number(score, "option_scores") for score in raw_scores]
            per_frame.append({"frame_id": item.get("frame_id", index), "scores": scores})
        if "option_margin" in item:
            margins.append(_finite_number(item["option_margin"], "option_margin"))
        elif scores and len(scores) >= 2:
            ordered = sorted(scores, reverse=True)
            margins.append(ordered[0] - ordered[1])

    retrieval_top_option: str | None = None
    if trace:
        ranked = sorted(
            enumerate(trace),
            key=lambda pair: (
                -float(pair[1].get("semantic_score", pair[1].get("objective_score", 0.0))),
                str(pair[1].get("frame_id", pair[0])),
            ),
        )
        best = ranked[0][1]
        best_index = best.get("best_option_index")
        if best_index is None:
            scores = best.get("option_scores")
            if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)) and scores:
                best_index = max(range(len(scores)), key=lambda index: float(scores[index]))
        if isinstance(best_index, int) and 0 <= best_index < 26:
            retrieval_top_option = chr(ord("A") + best_index)
    option_margin = max(margins) if margins else None
    return per_frame, [{"frame_id": item.get("frame_id"), "score": item.get("semantic_score")} for item in trace if "semantic_score" in item], option_margin, retrieval_top_option


def _memory_scores(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    snapshot = _frame_items(record.get("memory_snapshot"), "memory_snapshot")
    result = []
    for item in snapshot:
        scores = {
            key: _jsonable(item[key])
            for key in ("frame_id", "timestamp_s", "score", "semantic_score", "objective_score")
            if key in item
        }
        result.append(scores)
    return result


def _temporal_keywords(question: Any) -> list[str]:
    text = str(question).lower()
    return [keyword for keyword in _TEMPORAL_KEYWORDS if re.search(rf"\b{re.escape(keyword)}\b", text)]


def _has_timestamp(timestamps: Sequence[float], timestamp: float) -> bool:
    return any(abs(current - timestamp) <= _TIMESTAMP_TOLERANCE_S for current in timestamps)


def _metadata_check(
    recent: Mapping[str, Any], option: Mapping[str, Any]
) -> tuple[str, str, float, tuple[str, ...] | None]:
    for field in ("question", "ground_truth", "query_time_s"):
        if field not in recent or field not in option:
            raise ValueError(f"paired records require {field}")
    if recent["question"] != option["question"]:
        raise ValueError("question mismatch between paired records")
    if recent["ground_truth"] != option["ground_truth"]:
        raise ValueError("ground_truth mismatch between paired records")
    recent_query_time = _finite_number(recent["query_time_s"], "query_time_s")
    option_query_time = _finite_number(option["query_time_s"], "query_time_s")
    if abs(recent_query_time - option_query_time) > _TIMESTAMP_TOLERANCE_S:
        raise ValueError("query_time_s mismatch between paired records")
    recent_options = _record_options(recent)
    option_options = _record_options(option)
    if recent_options is not None and option_options is not None and recent_options != option_options:
        raise ValueError("options mismatch between paired records")
    if recent_options is not None and option_options is None:
        resolved_options = recent_options
    else:
        resolved_options = option_options
    return str(recent["question"]), str(recent["ground_truth"]), recent_query_time, resolved_options


def _is_correct(record: Mapping[str, Any], ground_truth: Any) -> bool:
    if "correct" in record and record["correct"] is not None:
        return bool(record["correct"])
    return record.get("prediction") == ground_truth


def _classify(recent_correct: bool, option_correct: bool) -> str:
    if recent_correct and option_correct:
        return "both_correct"
    if not recent_correct and option_correct:
        return "memory_helped"
    if recent_correct and not option_correct:
        return "memory_hurt"
    return "both_wrong"


def _group_rates(rows: Sequence[Mapping[str, Any]], predicate) -> dict[str, float]:
    rates: dict[str, float] = {}
    for group in _GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        rates[group] = (
            sum(bool(predicate(row)) for row in group_rows) / len(group_rows)
            if group_rows
            else 0.0
        )
    return rates


def _make_row(
    recent: Mapping[str, Any],
    option: Mapping[str, Any],
    *,
    recent_policy: str,
    option_policy: str,
) -> dict[str, Any]:
    question, ground_truth, query_time_s, options = _metadata_check(recent, option)
    _validate_causal_timestamps(recent, query_time_s)
    _validate_causal_timestamps(option, query_time_s)

    recent_frames = _record_frames(recent, memory=False)
    option_recent_frames = _record_frames(option, memory=False)
    option_memory_frames = _record_frames(option, memory=True)
    recent_timestamps = _timestamps(recent_frames)
    option_recent_timestamps = _timestamps(option_recent_frames)
    option_memory_timestamps = _timestamps(option_memory_frames)
    recent_correct = _is_correct(recent, ground_truth)
    option_correct = _is_correct(option, ground_truth)
    group = _classify(recent_correct, option_correct)

    trace = _traces(option)
    per_frame_option_scores, trace_memory_scores, option_margin, retrieval_top_option = _option_diagnostics(trace)
    memory_scores = _memory_scores(option)
    if not memory_scores:
        memory_scores = trace_memory_scores
    keywords = _temporal_keywords(question)
    lost_recent = [
        timestamp
        for timestamp in recent_timestamps
        if not _has_timestamp(option_recent_timestamps, timestamp)
    ]
    prediction = option.get("prediction")
    row: dict[str, Any] = {
        "sample_id": str(recent["sample_id"]),
        "video_id": str(recent["video_id"]),
        "video_index": int(recent["_video_index"]),
        "question": question,
        "ground_truth": str(ground_truth),
        "query_time_s": query_time_s,
        "options": list(options) if options is not None else None,
        "recent_policy": recent_policy,
        "option_policy": option_policy,
        "recent_prediction": _jsonable(recent.get("prediction")),
        "option_prediction": _jsonable(prediction),
        "recent_correct": recent_correct,
        "option_correct": option_correct,
        "group": group,
        "recent_timestamps": recent_timestamps,
        "recent_recent_timestamps": list(recent_timestamps),
        "option_recent_timestamps": option_recent_timestamps,
        "option_memory_timestamps": option_memory_timestamps,
        "recent_frame_temporal_distances": _temporal_distances(recent_timestamps, query_time_s),
        "option_recent_frame_temporal_distances": _temporal_distances(option_recent_timestamps, query_time_s),
        "option_memory_frame_temporal_distances": _temporal_distances(option_memory_timestamps, query_time_s),
        "recent_frame_distances_s": _temporal_distances(recent_timestamps, query_time_s),
        "option_recent_frame_distances_s": _temporal_distances(option_recent_timestamps, query_time_s),
        "option_memory_frame_distances_s": _temporal_distances(option_memory_timestamps, query_time_s),
        "option_semantic_trace": _jsonable(trace),
        "memory_scores": _jsonable(memory_scores),
        "per_frame_option_scores": _jsonable(per_frame_option_scores),
        "option_scores_by_frame": _jsonable(per_frame_option_scores),
        "option_margin": option_margin,
        "option_margins": [item.get("option_margin") for item in trace if "option_margin" in item],
        "retrieval_top_option": retrieval_top_option,
        "h1_recent_frame_loss_proxy": bool(lost_recent),
        "h1_recent_frame_loss_count": len(lost_recent),
        "h1_recent_frame_loss_rate": len(lost_recent) / len(recent_timestamps) if recent_timestamps else 0.0,
        "h2_retrieval_top_option_match": (
            retrieval_top_option == prediction
            if retrieval_top_option is not None and prediction is not None
            else None
        ),
        "temporal_keywords": keywords,
        "temporal_keyword_count": len(keywords),
        "temporal_keyword_present": bool(keywords),
        "h4_both_wrong_option_wrong": group == "both_wrong" and not option_correct,
    }
    return _jsonable(row)


def analyze_pair_records(
    recent_records: Iterable[Mapping[str, Any]],
    option_records: Iterable[Mapping[str, Any]],
    recent_policy: str,
    option_policy: str,
    min_video_index: int | None = None,
    max_video_index: int | None = None,
) -> PairAnalysisResult:
    """Pair two exact-policy record streams and summarize diagnosis groups."""
    if not isinstance(recent_policy, str) or not isinstance(option_policy, str):
        raise ValueError("policy IDs must be strings")
    _validate_range(min_video_index, "min_video_index")
    _validate_range(max_video_index, "max_video_index")
    if min_video_index is not None and max_video_index is not None and min_video_index > max_video_index:
        raise ValueError("min_video_index cannot exceed max_video_index")

    recent_by_id = _filtered_records(
        recent_records,
        recent_policy,
        min_video_index=min_video_index,
        max_video_index=max_video_index,
        label="recent",
    )
    option_by_id = _filtered_records(
        option_records,
        option_policy,
        min_video_index=min_video_index,
        max_video_index=max_video_index,
        label="option",
    )
    recent_ids = set(recent_by_id)
    option_ids = set(option_by_id)
    if recent_ids != option_ids:
        missing_from_option = sorted(recent_ids - option_ids)
        missing_from_recent = sorted(option_ids - recent_ids)
        raise ValueError(
            "missing paired records: "
            f"missing from option={missing_from_option}, missing from recent={missing_from_recent}"
        )

    rows = [
        _make_row(
            recent_by_id[sample_id],
            option_by_id[sample_id],
            recent_policy=recent_policy,
            option_policy=option_policy,
        )
        for sample_id in sorted(recent_ids)
    ]
    rows.sort(key=lambda row: (str(row["sample_id"]), int(row["video_index"]), str(row["video_id"])))

    n_pairs = len(rows)
    group_counts = {group: sum(row["group"] == group for row in rows) for group in _GROUPS}
    group_percentages = {
        group: (100.0 * group_counts[group] / n_pairs if n_pairs else 0.0)
        for group in _GROUPS
    }
    recent_accuracy = sum(bool(row["recent_correct"]) for row in rows) / n_pairs if n_pairs else 0.0
    option_accuracy = sum(bool(row["option_correct"]) for row in rows) / n_pairs if n_pairs else 0.0
    net_gain = group_counts["memory_helped"] - group_counts["memory_hurt"]
    accuracy_delta = option_accuracy - recent_accuracy
    summary: dict[str, Any] = {
        "n_pairs": n_pairs,
        "group_counts": group_counts,
        "group_percentages": group_percentages,
        "group_rates": {group: value / 100.0 for group, value in group_percentages.items()},
        "recent_accuracy": recent_accuracy,
        "option_accuracy": option_accuracy,
        "both_accuracies": {"recent": recent_accuracy, "option": option_accuracy},
        "net_gain": net_gain,
        "net_gain_rate": net_gain / n_pairs if n_pairs else 0.0,
        "helped_minus_hurt": net_gain,
        "accuracy_delta": accuracy_delta,
        "accuracy_delta_check": math.isclose(
            accuracy_delta,
            net_gain / n_pairs if n_pairs else 0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "h1_recent_frame_loss_rate_by_group": _group_rates(
            rows, lambda row: row["h1_recent_frame_loss_proxy"]
        ),
        "h2_retrieval_top_option_match_rate_by_group": _group_rates(
            rows, lambda row: row["h2_retrieval_top_option_match"] is True
        ),
        "h2_retrieval_top_option_match_rate": (
            sum(row["h2_retrieval_top_option_match"] is True for row in rows) / n_pairs
            if n_pairs
            else 0.0
        ),
        "h3_temporal_keyword_rates_by_group": _group_rates(
            rows, lambda row: row["temporal_keyword_present"]
        ),
        "h4_both_wrong_option_wrong_count": sum(
            row["h4_both_wrong_option_wrong"] for row in rows
        ),
        "h4_both_wrong_option_wrong_rate": (
            sum(row["h4_both_wrong_option_wrong"] for row in rows) / group_counts["both_wrong"]
            if group_counts["both_wrong"]
            else 0.0
        ),
    }
    return PairAnalysisResult(rows=rows, summary=_jsonable(summary))


def representative_rows(
    rows: Iterable[Mapping[str, Any]], group: str, limit: int = 20
) -> list[dict[str, Any]]:
    """Return at most ``limit`` stable, serializable rows from one group."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    candidates = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("rows must be mappings")
        if row.get("group") == group:
            candidate = _jsonable(dict(row))
            candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            str(row.get("sample_id", "")),
            int(row.get("video_index", 0)),
            str(row.get("video_id", "")),
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        )
    )
    return candidates[:limit]


def _artifact_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


_CSV_FIELDS = (
    "sample_id",
    "video_id",
    "video_index",
    "question",
    "ground_truth",
    "query_time_s",
    "recent_prediction",
    "option_prediction",
    "recent_correct",
    "option_correct",
    "group",
    "recent_timestamps",
    "recent_recent_timestamps",
    "option_recent_timestamps",
    "option_memory_timestamps",
    "recent_frame_temporal_distances",
    "option_recent_frame_temporal_distances",
    "option_memory_frame_temporal_distances",
    "option_semantic_trace",
    "memory_scores",
    "per_frame_option_scores",
    "option_margin",
    "retrieval_top_option",
    "h1_recent_frame_loss_proxy",
    "h2_retrieval_top_option_match",
    "temporal_keyword_present",
)


def write_pair_artifacts(
    result: PairAnalysisResult,
    output_prefix: str | Path,
    representative_limit: int = 20,
) -> dict[str, Path]:
    """Write CSV, summary JSON, and helped/hurt JSONL artifacts."""
    if not isinstance(result, PairAnalysisResult):
        raise TypeError("result must be a PairAnalysisResult")
    if isinstance(representative_limit, bool) or not isinstance(representative_limit, int) or representative_limit < 0:
        raise ValueError("representative_limit must be a non-negative integer")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": Path(f"{prefix}.csv"),
        "summary_json": Path(f"{prefix}-summary.json"),
        "helped_jsonl": Path(f"{prefix}-helped.jsonl"),
        "hurt_jsonl": Path(f"{prefix}-hurt.jsonl"),
    }
    rows = [_jsonable(dict(row)) for row in result.rows]
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _artifact_value(row.get(field)) for field in _CSV_FIELDS})
    paths["summary_json"].write_text(
        json.dumps(_jsonable(result.summary), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    for key, group in (("helped_jsonl", "memory_helped"), ("hurt_jsonl", "memory_hurt")):
        selected = representative_rows(rows, group, limit=representative_limit)
        with paths[key].open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return paths


__all__ = [
    "PairAnalysisResult",
    "analyze_pair_records",
    "representative_rows",
    "write_pair_artifacts",
]
