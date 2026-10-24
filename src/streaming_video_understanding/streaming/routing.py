"""Versioned question-only allocation rules; these are task proxies, not confidence."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AllocationRoute:
    group: str
    recent_capacity: int
    memory_capacity: int
    resolved_policy_id: str
    reason: str
    version: str = "question-rules-v1"


_COUNTING = re.compile(r"\b(?:how many times|number of times|total number of times)\b")
_HISTORY = re.compile(
    r"\b(?:before|after|earlier|previously|initially|originally|"
    r"first time|last time|at the beginning|at the start|so far)\b"
)
_CURRENT = re.compile(r"\b(?:now|currently|at the moment|at this moment|right now)\b")


def route_question(question: str, *, budget: int = 8) -> AllocationRoute:
    """Allocate B=8 without consulting options, labels, predictions or video frames.

    History cues take priority over current cues in mixed questions. Counting
    enables the existing temporal-diversity heuristic, not an event counter.
    Unknown questions conservatively use six recent and two historical frames.
    Rules are frozen as v1; change the public policy ID when modifying them.
    """
    if type(budget) is not int or budget != 8:
        raise ValueError("question-rules-v1 requires visual_budget=8")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    text = " ".join(question.casefold().split())
    if match := _COUNTING.search(text):
        return AllocationRoute("counting", 4, 4, "recent-4+semantic-diverse-memory-4", match.group())
    if match := _HISTORY.search(text):
        return AllocationRoute("history", 4, 4, "recent-4+semantic-memory-4", match.group())
    if match := _CURRENT.search(text):
        return AllocationRoute("current", 8, 0, "recent-8", match.group())
    return AllocationRoute("unspecified", 6, 2, "recent-6+semantic-memory-2", "no explicit temporal cue")
