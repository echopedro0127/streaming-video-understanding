from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.schema import StreamingSample


_ANSWER_LABELS = ("A", "B", "C", "D")
_REQUIRED_COVERAGE = (
    ("task_type", None, 1),
    ("frames_required", "multiple", 30),
    ("frames_required", "single", 20),
    ("temporal_clue_type", "Prior", 10),
    ("temporal_clue_type", "Concurrent", 20),
    ("answer", None, 1),
)


@dataclass(frozen=True)
class SplitResult:
    validation: tuple[StreamingSample, ...]
    final_test: tuple[StreamingSample, ...]
    stats: dict[str, Any]


def _stable_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def _field_value(sample: StreamingSample, field: str) -> str:
    value = getattr(sample, field)
    return str(value).strip()


def _metadata(sample: StreamingSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "task_type": sample.task_type,
        "frames_required": sample.frames_required,
        "temporal_clue_type": sample.temporal_clue_type,
        "ground_truth": sample.answer,
        "query_time_s": sample.query_time_s,
    }


def _stats(samples: tuple[StreamingSample, ...]) -> dict[str, Any]:
    return {
        "num_samples": len(samples),
        "num_videos": len({sample.video_id for sample in samples}),
        "task_type": dict(sorted(Counter(sample.task_type for sample in samples).items())),
        "frames_required": dict(
            sorted(Counter(sample.frames_required for sample in samples).items())
        ),
        "temporal_clue_type": dict(
            sorted(Counter(sample.temporal_clue_type for sample in samples).items())
        ),
        "ground_truth": dict(sorted(Counter(sample.answer for sample in samples).items())),
    }


def build_deterministic_split(
    samples: list[StreamingSample] | tuple[StreamingSample, ...],
    *,
    validation_size: int = 100,
    seed: int = 42,
) -> SplitResult:
    """Build a deterministic coverage-aware validation split and its complement."""
    if isinstance(validation_size, bool) or validation_size <= 0:
        raise ValueError("validation_size must be a positive integer")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    ordered = list(samples)
    if validation_size >= len(ordered):
        raise ValueError("validation_size must be smaller than the number of samples")
    sample_ids = [sample.sample_id for sample in ordered]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique")
    if any(not sample.frames_required for sample in ordered):
        raise ValueError("all annotations must provide frames_required for split coverage")

    ranked = sorted(ordered, key=lambda sample: _stable_rank(sample.sample_id, seed))
    known_task_types = {sample.task_type for sample in ordered}
    known_answers = set(_ANSWER_LABELS) & {sample.answer for sample in ordered}
    requirements: list[tuple[str, str | None, int, set[str]]] = []
    for field, value, minimum in _REQUIRED_COVERAGE:
        if field == "task_type":
            values = known_task_types
            effective_minimum = minimum
        elif field == "answer":
            values = known_answers
            effective_minimum = minimum
        else:
            values = {value} if value is not None else set()
            effective_minimum = minimum
        for required_value in sorted(values):
            requirements.append((field, required_value, effective_minimum, {required_value}))

    selected: list[StreamingSample] = []
    selected_ids: set[str] = set()
    counts: Counter[str] = Counter()
    while len(selected) < validation_size:
        deficits = {
            (field, value): minimum - counts[f"{field}={value}"]
            for field, value, minimum, _ in requirements
            if counts[f"{field}={value}"] < minimum
        }
        if not deficits:
            break
        candidates = [sample for sample in ranked if sample.sample_id not in selected_ids]
        if not candidates:
            break

        def score(sample: StreamingSample) -> tuple[int, int, str]:
            gain = sum(
                max(0, deficits.get((field, value), 0))
                for field, value, _, _ in requirements
                if _field_value(sample, field) == value
            )
            rank = ranked.index(sample)
            return gain, -rank, sample.sample_id

        best = max(candidates, key=score)
        selected.append(best)
        selected_ids.add(best.sample_id)
        for field, value, _, _ in requirements:
            if _field_value(best, field) == value:
                counts[f"{field}={value}"] += 1

    if len(selected) < validation_size:
        selected_ids.update(sample.sample_id for sample in selected)
        for sample in ranked:
            if len(selected) >= validation_size:
                break
            if sample.sample_id not in selected_ids:
                selected.append(sample)
                selected_ids.add(sample.sample_id)

    missing = [
        f"{field}={value} (need {minimum}, got {counts[f'{field}={value}']})"
        for field, value, minimum, _ in requirements
        if counts[f"{field}={value}"] < minimum
    ]
    if len(selected) != validation_size or missing:
        raise ValueError(
            "could not construct validation split: "
            + "; ".join(missing or [f"selected {len(selected)} of {validation_size}"])
        )

    validation_ids = {sample.sample_id for sample in selected}
    final_test = tuple(sample for sample in ordered if sample.sample_id not in validation_ids)
    validation = tuple(selected)
    stats = {
        "seed": seed,
        "validation": _stats(validation),
        "final_test": _stats(final_test),
        "total": _stats(tuple(ordered)),
    }
    return SplitResult(validation=validation, final_test=final_test, stats=stats)


def write_split_files(split: SplitResult, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    validation_path = output_path / "validation.json"
    final_test_path = output_path / "final_test.json"
    stats_path = output_path / "stats.json"
    validation_path.write_text(
        json.dumps(
            {
                "split": "validation",
                "sample_ids": [sample.sample_id for sample in split.validation],
                "samples": [_metadata(sample) for sample in split.validation],
                "stats": split.stats["validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    final_test_path.write_text(
        json.dumps(
            {
                "split": "final_test",
                "sample_ids": [sample.sample_id for sample in split.final_test],
                "samples": [_metadata(sample) for sample in split.final_test],
                "stats": split.stats["final_test"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(split.stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"validation": validation_path, "final_test": final_test_path, "stats": stats_path}


def load_split_sample_ids(path: str | Path) -> set[str]:
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"split file does not exist: {split_path}")
    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed split JSON: {split_path}") from exc
    sample_ids = payload.get("sample_ids") if isinstance(payload, dict) else None
    if not isinstance(sample_ids, list) or not all(isinstance(item, str) for item in sample_ids):
        raise ValueError(f"split file must contain a string sample_ids list: {split_path}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"split file contains duplicate sample IDs: {split_path}")
    return set(sample_ids)
