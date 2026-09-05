from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..data.schema import FrameRecord


class FutureFrameLeakageError(ValueError):
    """Raised when a frame newer than the visible query boundary is used."""


@dataclass
class CausalFrameBuffer:
    visible_until_s: float
    context_start_s: float = 0.0
    timestamp_tolerance_s: float = 1e-6
    _frames: list[FrameRecord] = field(default_factory=list, init=False, repr=False)

    def add(self, frame: FrameRecord) -> None:
        if not math.isfinite(frame.timestamp_s) or frame.timestamp_s < 0:
            raise ValueError(f"invalid frame timestamp: {frame.timestamp_s!r}")
        if frame.timestamp_s > self.visible_until_s + self.timestamp_tolerance_s:
            raise FutureFrameLeakageError(
                f"Frame {frame.frame_id} at {frame.timestamp_s:.6f}s is after "
                f"visible time {self.visible_until_s:.6f}s"
            )
        self._frames.append(frame)

    def snapshot(self) -> tuple[FrameRecord, ...]:
        return tuple(
            sorted(
                (
                    frame
                    for frame in self._frames
                    if self.context_start_s - self.timestamp_tolerance_s
                    <= frame.timestamp_s
                    <= self.visible_until_s + self.timestamp_tolerance_s
                ),
                key=lambda frame: (frame.timestamp_s, frame.frame_id),
            )
        )
