from __future__ import annotations

import ast
import csv
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

from .schema import StreamingSample, parse_timestamp


_OPTION_PREFIX = re.compile(r"^\s*[A-D][.)]\s*")
_REQUIRED_COLUMNS = {
    "question_id",
    "task_type",
    "question",
    "time_stamp",
    "answer",
    "options",
    "video",
}


def _parse_options(value: object) -> tuple[str, ...]:
    try:
        parsed = value if isinstance(value, (list, tuple)) else ast.literal_eval(str(value))
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid options field: {value!r}") from exc
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError("Options must be a non-empty list")
    return tuple(_OPTION_PREFIX.sub("", str(option)).strip() for option in parsed)


def _normalize_answer(value: str) -> str:
    answer = str(value).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"StreamingBench answer must be A-D, got {value!r}")
    return answer


def _required_columns(fieldnames: Iterable[str | None]) -> None:
    columns = {field for field in fieldnames if field}
    missing = sorted(_REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"Missing required StreamingBench columns: {', '.join(missing)}")


def _rows_to_samples(rows: Iterable[Mapping[str, object]], *, source: Path, start_line: int = 1) -> list[StreamingSample]:
    rows = list(rows)
    _required_columns(rows[0].keys() if rows else [])
    samples: list[StreamingSample] = []
    for offset, row in enumerate(rows):
        line_number = start_line + offset
        try:
            sample_id = str(row["question_id"] or "").strip()
            video_text = str(row["video"] or "").strip()
            question = str(row["question"] or "").strip()
            task_type = str(row["task_type"] or "").strip()
            if not sample_id or not video_text or not question or not task_type:
                raise ValueError("question_id, video, question, and task_type must be non-empty")
            video_path = Path(video_text)
            video_id = (
                video_path.parent.name
                if video_path.stem.lower() == "video" and video_path.parent.name
                else video_path.stem
            )
            samples.append(
                StreamingSample(
                    sample_id=sample_id,
                    video_id=video_id,
                    video_path=video_path,
                    question=question,
                    options=_parse_options(row["options"]),
                    answer=_normalize_answer(str(row["answer"])),
                    query_time_s=parse_timestamp(str(row["time_stamp"])),
                    task_type=task_type,
                    temporal_clue_type=str(row.get("temporal_clue_type") or "").strip(),
                    frames_required=str(row.get("frames_required") or "").strip().lower(),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid StreamingBench row {line_number} in {source}: {exc}") from exc
    return samples


def load_streamingbench_csv(path: str | Path) -> list[StreamingSample]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"StreamingBench annotation file does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return _rows_to_samples(reader, source=csv_path, start_line=2)


def load_streamingbench_file(path: str | Path) -> list[StreamingSample]:
    annotation_path = Path(path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"StreamingBench annotation file does not exist: {annotation_path}")
    suffix = annotation_path.suffix.lower()
    if suffix == ".csv":
        return load_streamingbench_csv(annotation_path)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in annotation_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError(f"Expected a list of annotation objects in {annotation_path}")
        return _rows_to_samples(rows, source=annotation_path)
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Parquet annotations require pandas and a parquet engine; install requirements.txt"
            ) from exc
        frame = pd.read_parquet(annotation_path)
        return _rows_to_samples(frame.to_dict(orient="records"), source=annotation_path)
    raise ValueError(f"Unsupported annotation format: {annotation_path.suffix}")
