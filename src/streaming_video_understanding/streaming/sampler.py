from __future__ import annotations

from collections.abc import Sequence

from ..data.schema import FrameRecord
from .buffer import FutureFrameLeakageError


def _causal_frames(
    frames: Sequence[FrameRecord],
    visible_until_s: float,
    tolerance_s: float,
) -> tuple[FrameRecord, ...]:
    ordered = tuple(sorted(frames, key=lambda frame: (frame.timestamp_s, frame.frame_id)))
    future = [frame for frame in ordered if frame.timestamp_s > visible_until_s + tolerance_s]
    if future:
        first = future[0]
        raise FutureFrameLeakageError(
            f"Frame {first.frame_id} at {first.timestamp_s:.6f}s is after "
            f"visible time {visible_until_s:.6f}s"
        )
    return tuple(frame for frame in ordered if frame.timestamp_s <= visible_until_s + tolerance_s)


def _uniform_select(frames: Sequence[FrameRecord], capacity: int) -> tuple[FrameRecord, ...]:
    ordered = tuple(frames)
    if capacity <= 0 or not ordered:
        return ()
    if len(ordered) <= capacity:
        return ordered
    indexes = [int(index * (len(ordered) - 1) / (capacity - 1)) for index in range(capacity)]
    return tuple(ordered[index] for index in indexes)


def recent_window(
    frames: Sequence[FrameRecord],
    *,
    visible_until_s: float,
    capacity: int,
    tolerance_s: float = 1e-6,
) -> tuple[FrameRecord, ...]:
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    visible = _causal_frames(frames, visible_until_s, tolerance_s)
    return visible[-capacity:] if capacity else ()


def uniform_history(
    frames: Sequence[FrameRecord],
    *,
    visible_until_s: float,
    capacity: int,
    tolerance_s: float = 1e-6,
) -> tuple[FrameRecord, ...]:
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    visible = _causal_frames(frames, visible_until_s, tolerance_s)
    return _uniform_select(visible, capacity)


def recent_plus_memory(
    frames: Sequence[FrameRecord],
    *,
    visible_until_s: float,
    recent_capacity: int,
    memory_capacity: int,
    tolerance_s: float = 1e-6,
) -> tuple[FrameRecord, ...]:
    if recent_capacity < 0 or memory_capacity < 0:
        raise ValueError("recent_capacity and memory_capacity must be non-negative")
    visible = _causal_frames(frames, visible_until_s, tolerance_s)
    recent = visible[-recent_capacity:] if recent_capacity else ()
    history = visible[:-recent_capacity] if recent_capacity else visible
    memory = _uniform_select(history, memory_capacity)
    return tuple(sorted((*memory, *recent), key=lambda frame: (frame.timestamp_s, frame.frame_id)))


def timestamp_aware_memory(
    frames: Sequence[FrameRecord],
    *,
    visible_until_s: float,
    recent_capacity: int,
    memory_capacity: int,
    tolerance_s: float = 1e-6,
) -> tuple[FrameRecord, ...]:
    return recent_plus_memory(
        frames,
        visible_until_s=visible_until_s,
        recent_capacity=recent_capacity,
        memory_capacity=memory_capacity,
        tolerance_s=tolerance_s,
    )
