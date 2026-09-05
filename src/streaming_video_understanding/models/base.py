from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class ModelRequest:
    images: tuple[Image.Image, ...]
    timestamps_s: tuple[float, ...]
    question: str
    options: tuple[str, ...]
    query_time_s: float
    include_timestamps: bool


@dataclass(frozen=True)
class ModelResponse:
    raw_text: str
    answer: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelBackend(Protocol):
    def infer(self, request: ModelRequest) -> ModelResponse:
        """Run one deterministic single-sample inference request."""
