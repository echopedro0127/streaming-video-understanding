"""Post-hoc option-order diagnostics with frozen visual inputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .pairs import _frame_items, _record_options, _validate_causal_timestamps, _prompt_options, _OPTION_LINE_RE


def _label(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or value not in ("A", "B", "C", "D"):
        raise ValueError(f"{field} must be an exact A-D label")
    return value


def _validated(row: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...], float]:
    prediction, truth = _label(row, "prediction"), _label(row, "ground_truth")
    if type(row.get("correct")) is not bool or row["correct"] != (prediction == truth):
        raise ValueError("correct disagrees with prediction and ground_truth")
    options = _record_options(row)
    if options is None or len(options) != 4:
        raise ValueError("four original/permuted options are required")
    if len({text.strip().casefold() for text in options}) != 4:
        raise ValueError("ambiguous duplicate option text")
    prompt = row.get("timestamp_prompt")
    if not isinstance(prompt, str) or _prompt_options(prompt) != options:
        raise ValueError("actual prompt options disagree with recorded options")
    time = row.get("query_time_s")
    if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0:
        raise ValueError("query_time_s must be finite and non-negative")
    _validate_causal_timestamps(row, float(time))
    timestamps = row.get("timestamps_s")
    if (not isinstance(timestamps, (list, tuple))
        or tuple(timestamps) != tuple(timestamp for _, timestamp in _manifest(row))):
        raise ValueError("ordered timestamps_s disagree with selected frame manifest")
    return prediction, truth, options, float(time)


def _manifest(row: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    raw = row.get("selected_frames")
    if not isinstance(raw, list):
        raise ValueError("selected_frames must be a list")
    _frame_items(raw, "selected_frames")  # validation including duplicate IDs
    # Preserve actual image input order; sorting here would hide a confound.
    return tuple((frame["frame_id"], float(frame["timestamp_s"])) for frame in raw)


def _comparison(original: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    old_pred, old_truth, options, time = _validated(original)
    new_pred, new_truth, new_options, new_time = _validated(candidate)
    config_id = original.get("config_id")
    if (not isinstance(config_id, str) or not config_id
        or "option_permutation_seed" in original
        or candidate.get("config_id") != f"{config_id}-option-permutation-{candidate['option_permutation_seed']}"):
        raise ValueError("configuration mismatch; require same-config unpermuted control")
    def masked_prompt(record):
        return "\n".join(_OPTION_LINE_RE.sub(r"\1. <OPTION>", line)
                         for line in record["timestamp_prompt"].splitlines())
    if masked_prompt(original) != masked_prompt(candidate):
        raise ValueError("prompt instructions/content changed beyond option order")
    if any(original.get(key) != candidate.get(key) for key in ("question", "video_id")):
        raise ValueError("question/video mismatch")
    if abs(time - new_time) > 1e-6:
        raise ValueError("query timestamp mismatch")
    if original.get("sample_fps") != candidate.get("sample_fps"):
        raise ValueError("sample FPS mismatch")
    if _manifest(original) != _manifest(candidate):
        raise ValueError("selected frames changed; option-order attribution is invalid")
    permutation = candidate.get("option_permutation")
    if (not isinstance(permutation, (list, tuple)) or len(permutation) != 4
        or any(type(index) is not int for index in permutation)
        or set(permutation) != {0, 1, 2, 3}):
        raise ValueError("option_permutation must be a permutation of 0..3")
    if tuple(candidate.get("original_options", ())) != options:
        raise ValueError("original_options mismatch")
    expected = tuple(options[index] for index in permutation)
    if new_options != expected or tuple(candidate.get("permuted_options", ())) != expected:
        raise ValueError("permuted options disagree with permutation")
    if candidate.get("original_ground_truth") != old_truth:
        raise ValueError("original ground truth mismatch")
    if new_truth != chr(65 + permutation.index(ord(old_truth) - 65)):
        raise ValueError("ground truth remapping mismatch")
    mapped = chr(65 + permutation[ord(new_pred) - 65])
    opportunity = permutation[ord(old_pred) - 65] != ord(old_pred) - 65
    return {
        "sample_id": original["sample_id"], "video_id": original["video_id"],
        "policy_id": original["policy_id"], "seed": candidate["option_permutation_seed"],
        "query_time_s": time, "ground_truth": old_truth,
        "original_prediction": old_pred, "permuted_prediction": new_pred,
        "mapped_prediction": mapped, "predicted_content": options[ord(mapped) - 65],
        "original_predicted_content": options[ord(old_pred) - 65],
        "original_correct": original["correct"], "permuted_correct": candidate["correct"],
        "semantic_consistent": mapped == old_pred,
        "label_sticky_opportunity": opportunity,
        "label_sticky": opportunity and old_pred == new_pred,
        "selected_frames_match": True,
        "permutation": list(permutation),
        "original_config_id": original.get("config_id"),
        "permuted_config_id": candidate.get("config_id"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    opportunities = sum(row["label_sticky_opportunity"] for row in rows)
    sticky = sum(row["label_sticky"] for row in rows)
    consistent = sum(row["semantic_consistent"] for row in rows)
    return {
        "n_comparisons": n, "semantic_consistent_count": consistent,
        "semantic_consistency": consistent / n,
        "original_accuracy": sum(row["original_correct"] for row in rows) / n,
        "permuted_accuracy": sum(row["permuted_correct"] for row in rows) / n,
        "correct_to_wrong": sum(row["original_correct"] and not row["permuted_correct"] for row in rows),
        "wrong_to_correct": sum(not row["original_correct"] and row["permuted_correct"] for row in rows),
        "label_sticky_opportunities": opportunities, "label_sticky_count": sticky,
        "label_sticky_rate": sticky / opportunities if opportunities else None,
    }


def analyze_option_stability(
    original_records: Sequence[Mapping[str, Any]], permuted_records: Sequence[Mapping[str, Any]],
    *, sample_ids: Sequence[str], policies: Sequence[str], seeds: Sequence[int],
) -> dict[str, Any]:
    """Require a complete sample x policy x seed matrix and compare answer content."""
    for values, name in ((sample_ids, "sample_ids"), (policies, "policies")):
        if (isinstance(values, (str, bytes)) or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))):
            raise ValueError(f"{name} must be distinct non-empty strings")
    if not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be distinct integers")
    expected_original = {(sample, policy) for sample in sample_ids for policy in policies}
    expected = {(sample, policy, seed) for sample, policy in expected_original for seed in seeds}
    originals, candidates = {}, {}
    for row in original_records:
        key = (row.get("sample_id"), row.get("policy_id"))
        if key not in expected_original:
            continue
        if key in originals:
            raise ValueError(f"duplicate original key {key}")
        originals[key] = row
    for row in permuted_records:
        seed = row.get("option_permutation_seed")
        if type(seed) is not int:
            raise ValueError("option_permutation_seed must be an integer")
        key = (row.get("sample_id"), row.get("policy_id"), seed)
        if key not in expected:
            raise ValueError(f"unexpected permuted key {key}")
        if key in candidates:
            raise ValueError(f"duplicate permuted key {key}")
        candidates[key] = row
    if set(originals) != expected_original or set(candidates) != expected:
        raise ValueError("missing original or permuted records in expected matrix")
    rows = [_comparison(originals[sample, policy], candidates[sample, policy, seed])
            for sample, policy, seed in sorted(expected)]
    conditions = []
    for policy in sorted(policies):
        for seed in sorted(seeds):
            selected = [row for row in rows if row["policy_id"] == policy and row["seed"] == seed]
            conditions.append({"policy_id": policy, "seed": seed, **_aggregate(selected)})
    return {"rows": rows, "summary": {
        "n_samples": len(sample_ids), "n_comparisons": len(rows),
        "seeds": sorted(seeds), "conditions": conditions,
        "by_policy": {policy: _aggregate([row for row in rows if row["policy_id"] == policy])
                      for policy in sorted(policies)},
        "scope": "Post-hoc outcome-selected diagnostic; not representative benchmark accuracy or calibrated confidence.",
        "control": "Selected frame manifests and order match the stored original condition for every comparison.",
    }}
