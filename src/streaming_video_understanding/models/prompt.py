from __future__ import annotations

import re


def build_prompt(
    *,
    question: str,
    options: tuple[str, ...],
    timestamps_s: tuple[float, ...],
    query_time_s: float,
    include_timestamps: bool,
) -> str:
    if any(timestamp < 0 for timestamp in timestamps_s):
        raise ValueError("timestamps_s must be non-negative")
    if any(
        current > next_timestamp
        for current, next_timestamp in zip(timestamps_s, timestamps_s[1:])
    ):
        raise ValueError("timestamps_s must be chronological")
    lines = [
        "Answer the following multiple-choice video question.",
        "Answer with only one letter: A, B, C, or D.",
    ]
    if include_timestamps:
        lines.append("The ordered visual frames have these timestamps:")
        for index, timestamp_s in enumerate(timestamps_s, start=1):
            lines.append(
                f"[frame {index} | absolute_time={timestamp_s:.2f}s | "
                f"relative_to_query={timestamp_s - query_time_s:.2f}s]"
            )
    lines.append(f"Question: {question.strip()}")
    lines.append("Options:")
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        lines.append(f"{letter}. {option.strip()}")
    return "\n".join(lines)


def parse_answer_letter(raw_text: str) -> str:
    candidates = re.findall(r"(?<![A-Z])([A-D])(?![A-Z])", raw_text.upper())
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise ValueError(f"Could not parse an answer letter from model output: {raw_text!r}")
    if len(unique) > 1:
        raise ValueError(f"ambiguous answer output: {raw_text!r}")
    return unique[0]
