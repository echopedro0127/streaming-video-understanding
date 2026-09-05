from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ResumeStore:
    """Durable JSONL completion index keyed by sample and experiment config."""

    def __init__(self, path: str | Path, *, repair: bool = False) -> None:
        self.path = Path(path)
        self.completed_keys: set[tuple[str, str]] = set()
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if repair:
                        continue
                    raise ValueError(
                        f"Malformed JSONL record at {self.path}, line {line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    if repair:
                        continue
                    raise ValueError(
                        f"JSONL record at {self.path}, line {line_number} must be an object"
                    )
                sample_id = record.get("sample_id")
                config_id = record.get("config_id")
                if sample_id is not None and config_id is not None:
                    self.completed_keys.add((str(sample_id), str(config_id)))

    def is_complete(self, sample_id: str, config_id: str) -> bool:
        return (str(sample_id), str(config_id)) in self.completed_keys

    def append(self, record: dict[str, Any]) -> None:
        sample_id = record.get("sample_id")
        config_id = record.get("config_id")
        if sample_id is None or config_id is None:
            raise ValueError("resume records require sample_id and config_id")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        self.completed_keys.add((str(sample_id), str(config_id)))
