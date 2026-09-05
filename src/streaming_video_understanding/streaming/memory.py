from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..data.schema import FrameRecord
from .buffer import FutureFrameLeakageError


@dataclass
class BoundedTemporalMemory:
    capacity: int
    visible_until_s: float
    timestamp_tolerance_s: float = 1e-6
    _frames: list[FrameRecord] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError("memory capacity must be non-negative")

    def update(self, frame: FrameRecord) -> None:
        if not math.isfinite(frame.timestamp_s) or frame.timestamp_s < 0:
            raise ValueError(f"invalid frame timestamp: {frame.timestamp_s!r}")
        if frame.timestamp_s > self.visible_until_s + self.timestamp_tolerance_s:
            raise FutureFrameLeakageError(
                f"Frame {frame.frame_id} at {frame.timestamp_s:.6f}s is after "
                f"visible time {self.visible_until_s:.6f}s"
            )
        if self.capacity == 0:
            return
        self._frames.append(frame)
        self._frames = list(_uniform_select(self._frames, self.capacity))

    def snapshot(self) -> tuple[FrameRecord, ...]:
        return tuple(self._frames)


def _uniform_select(frames: list[FrameRecord], capacity: int) -> tuple[FrameRecord, ...]:
    ordered = tuple(sorted(frames, key=lambda frame: (frame.timestamp_s, frame.frame_id)))
    if capacity <= 0 or not ordered:
        return ()
    if len(ordered) <= capacity:
        return ordered
    if capacity == 1:
        return (ordered[0],)
    indexes = [int(index * (len(ordered) - 1) / (capacity - 1)) for index in range(capacity)]
    return tuple(ordered[index] for index in indexes)
