from __future__ import annotations

import math
from collections.abc import Sequence

from ..data.schema import FrameRecord
from .buffer import FutureFrameLeakageError


def _causal_frames(
    frames: Sequence[FrameRecord],
    visible_until_s: float,
    tolerance_s: float,
) -> tuple[FrameRecord, ...]:
    for frame in frames:
        if not math.isfinite(frame.timestamp_s) or frame.timestamp_s < 0:
            raise ValueError(f"invalid frame timestamp: {frame.timestamp_s!r}")
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
    if capacity == 1:
        return (ordered[0],)
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


def timestamp_aware_history(
    frames: Sequence[FrameRecord],
    *,
    visible_until_s: float,
    capacity: int,
    tolerance_s: float = 1e-6,
) -> tuple[FrameRecord, ...]:
    """Select historical frames near evenly spaced temporal-bin centers."""
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    visible = _causal_frames(frames, visible_until_s, tolerance_s)
    if capacity <= 0 or not visible:
        return ()
    if len(visible) <= capacity:
        return visible

    start_s = visible[0].timestamp_s
    end_s = visible[-1].timestamp_s
    if end_s <= start_s:
        return _uniform_select(visible, capacity)

    selected: list[FrameRecord] = []
    for index in range(capacity):
        center_s = start_s + (index + 0.5) * (end_s - start_s) / capacity
        candidate = min(
            (frame for frame in visible if frame not in selected),
            key=lambda frame: (
                abs(frame.timestamp_s - center_s),
                frame.timestamp_s,
                frame.frame_id,
            ),
        )
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda frame: (frame.timestamp_s, frame.frame_id)))


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
    if recent_capacity < 0 or memory_capacity < 0:
        raise ValueError("recent_capacity and memory_capacity must be non-negative")
    visible = _causal_frames(frames, visible_until_s, tolerance_s)
    recent = visible[-recent_capacity:] if recent_capacity else ()
    history = visible[:-recent_capacity] if recent_capacity else visible
    memory = timestamp_aware_history(
        history,
        visible_until_s=visible_until_s,
        capacity=memory_capacity,
        tolerance_s=tolerance_s,
    )
    return tuple(sorted((*memory, *recent), key=lambda frame: (frame.timestamp_s, frame.frame_id)))
