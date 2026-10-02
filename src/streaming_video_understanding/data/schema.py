from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp_s: float


@dataclass(frozen=True)
class StreamingSample:
    sample_id: str
    video_id: str
    video_path: Path
    question: str
    options: tuple[str, ...]
    answer: str
    query_time_s: float
    task_type: str
    temporal_clue_type: str
    frames_required: str = ""


def parse_timestamp(value: str | int | float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Invalid timestamp: empty value")
        try:
            parts = text.split(":")
            if len(parts) == 1:
                seconds = float(parts[0])
            elif len(parts) == 2:
                minutes, remainder = parts
                seconds = float(minutes) * 60.0 + float(remainder)
            elif len(parts) == 3:
                hours, minutes, remainder = parts
                seconds = float(hours) * 3600.0 + float(minutes) * 60.0 + float(remainder)
            else:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid timestamp: {value!r}") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return seconds
