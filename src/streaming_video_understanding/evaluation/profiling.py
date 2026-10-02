from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class RuntimeProfileStore:
    """Append-only sidecar store for per-sample stage-level timings."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        if not record.get("sample_id"):
            raise ValueError("runtime profile requires sample_id")
        if not record.get("config_id"):
            raise ValueError("runtime profile requires config_id")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
