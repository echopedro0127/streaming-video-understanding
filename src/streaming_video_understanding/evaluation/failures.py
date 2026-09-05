from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FailureStore:
    """Append-only, JSON-serializable evidence for incorrect predictions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        if record.get("correct"):
            return
        failed_record = dict(record)
        failed_record.setdefault("status", "incorrect")
        self._append(failed_record)

    def append_failure(self, record: dict[str, Any]) -> None:
        failed_record = dict(record)
        failed_record["correct"] = False
        failed_record.setdefault("status", "failed")
        self._append(failed_record)

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
