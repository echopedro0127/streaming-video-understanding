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
    readout_mode: str = "mcq"
    allow_invalid_answer: bool = False


@dataclass(frozen=True)
class ModelResponse:
    raw_text: str
    answer: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    processor_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    postprocess_latency_ms: float | None = None
    predicted_count: int | None = None
    parse_valid: bool = True


class ModelBackend(Protocol):
    def infer(self, request: ModelRequest) -> ModelResponse:
        """Run one deterministic single-sample inference request."""
