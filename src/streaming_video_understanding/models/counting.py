"""Frozen choice-free count format; no options or ground truth enter its prompt."""
from __future__ import annotations

import re
from collections.abc import Sequence


def parse_count(text: str) -> int | None:
    match = re.fullmatch(r"\s*([0-9]+)(?:\s+times?)?\s*[.!]?\s*", text)
    return int(match.group(1)) if match else None


def numeric_options(options: Sequence[str]) -> tuple[int, ...]:
    if len(options) != 4:
        raise ValueError("count readout requires four numeric options")
    result = []
    for option in options:
        if not isinstance(option, str) or not re.fullmatch(r"\s*[0-9]+\.?\s*", option):
            raise ValueError("count readout requires non-negative integer option text")
        result.append(int(option.strip().rstrip(".")))
    if len(set(result)) != 4:
        raise ValueError("numeric option values must be distinct")
    return tuple(result)


def build_count_prompt(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("count question must be non-empty")
    return ("Count the events asked about in the video.\n"
            "Answer with one non-negative integer only.\n"
            f"Question: {question.strip()}")
