from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from ..data.schema import StreamingSample


@dataclass(frozen=True)
class PermutationResult:
    sample: StreamingSample
    original_answer: str
    permutation: tuple[int, ...]


def _option_rank(seed: int, sample_id: str, index: int) -> str:
    return hashlib.sha256(
        f"{seed}:{sample_id}:option:{index}".encode("utf-8")
    ).hexdigest()


def permute_sample(sample: StreamingSample, *, seed: int) -> PermutationResult:
    """Return a deterministically permuted copy with a remapped answer label."""
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if len(sample.options) != 4:
        raise ValueError("option permutation requires exactly four options")
    if sample.answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"invalid answer label: {sample.answer!r}")

    permutation = tuple(
        sorted(
            range(len(sample.options)),
            key=lambda index: _option_rank(seed, sample.sample_id, index),
        )
    )
    if permutation == tuple(range(len(sample.options))):
        permutation = (1, 0, 2, 3)
    original_index = ord(sample.answer) - ord("A")
    new_index = permutation.index(original_index)
    permuted = replace(
        sample,
        options=tuple(sample.options[index] for index in permutation),
        answer=chr(ord("A") + new_index),
    )
    return PermutationResult(
        sample=permuted,
        original_answer=sample.answer,
        permutation=permutation,
    )
