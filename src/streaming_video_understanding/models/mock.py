from __future__ import annotations

import time

from .base import ModelRequest, ModelResponse


class MockVLMBackend:
    def __init__(self, answer: str = "A") -> None:
        normalized = answer.strip().upper()
        if normalized not in {"A", "B", "C", "D"}:
            raise ValueError("Mock answer must be one of A, B, C, or D")
        self.answer = normalized
        self.call_count = 0
        self.last_request: ModelRequest | None = None

    def infer(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        self.call_count += 1
        self.last_request = request
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ModelResponse(
            raw_text=self.answer,
            answer=self.answer,
            latency_ms=latency_ms,
        )
