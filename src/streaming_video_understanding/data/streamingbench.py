from __future__ import annotations

import ast
import csv
import re
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


def _parse_options(value: str) -> tuple[str, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
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


def load_streamingbench_csv(path: str | Path) -> list[StreamingSample]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"StreamingBench annotation file does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _required_columns(reader.fieldnames or [])
        rows: list[StreamingSample] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                sample_id = str(row["question_id"]).strip()
                video_path = Path(str(row["video"]).strip())
                rows.append(
                    StreamingSample(
                        sample_id=sample_id,
                        video_id=video_path.stem,
                        video_path=video_path,
                        question=str(row["question"]).strip(),
                        options=_parse_options(str(row["options"])),
                        answer=_normalize_answer(str(row["answer"])),
                        query_time_s=parse_timestamp(str(row["time_stamp"])),
                        task_type=str(row["task_type"]).strip(),
                        temporal_clue_type=str(row.get("temporal_clue_type") or "").strip(),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid StreamingBench row {line_number}: {exc}") from exc
    return rows
