import pytest

from streaming_video_understanding.data.schema import FrameRecord
from streaming_video_understanding.streaming.buffer import (
    CausalFrameBuffer,
    FutureFrameLeakageError,
)
from streaming_video_understanding.streaming.memory import BoundedTemporalMemory
from streaming_video_understanding.streaming.sampler import (
    recent_plus_memory,
    recent_window,
    timestamp_aware_memory,
    uniform_history,
)


def frames():
    return tuple(FrameRecord(i, float(i * 10)) for i in range(7))


def test_buffer_rejects_frame_after_visible_time():
    buffer = CausalFrameBuffer(visible_until_s=30.0)
    with pytest.raises(FutureFrameLeakageError):
        buffer.add(FrameRecord(4, 40.0))


def test_buffer_snapshot_respects_context_start_and_order():
    buffer = CausalFrameBuffer(visible_until_s=50.0, context_start_s=15.0)
    for frame in frames():
        if frame.timestamp_s <= 50.0:
            buffer.add(frame)

    assert [frame.frame_id for frame in buffer.snapshot()] == [2, 3, 4, 5]


def test_recent_window_returns_latest_visible_frames():
    selected = recent_window(frames()[:-1], visible_until_s=50.0, capacity=3)
    assert [frame.frame_id for frame in selected] == [3, 4, 5]


def test_uniform_history_is_deterministic_and_causal():
    selected = uniform_history(frames()[:-1], visible_until_s=50.0, capacity=4)
    assert [frame.frame_id for frame in selected] == [0, 1, 3, 5]
    assert max(frame.timestamp_s for frame in selected) <= 50.0


def test_uniform_history_supports_single_slot():
    selected = uniform_history(frames()[:-1], visible_until_s=50.0, capacity=1)
    assert len(selected) == 1


def test_buffer_rejects_non_finite_or_negative_timestamp():
    buffer = CausalFrameBuffer(visible_until_s=50.0)
    with pytest.raises(ValueError, match="timestamp"):
        buffer.add(FrameRecord(7, float("nan")))
    with pytest.raises(ValueError, match="timestamp"):
        buffer.add(FrameRecord(8, -1.0))


def test_bounded_memory_never_exceeds_capacity():
    memory = BoundedTemporalMemory(capacity=3, visible_until_s=60.0)
    for frame in frames():
        memory.update(frame)

    selected = memory.snapshot()
    assert len(selected) == 3
    assert [frame.timestamp_s for frame in selected] == sorted(frame.timestamp_s for frame in selected)


def test_recent_plus_memory_has_fixed_total_budget():
    selected = recent_plus_memory(
        frames()[:-1], visible_until_s=50.0, recent_capacity=2, memory_capacity=2
    )
    assert len(selected) == 4
    assert [frame.frame_id for frame in selected] == [0, 3, 4, 5]


def test_timestamp_aware_memory_keeps_same_frame_selection():
    ordinary = recent_plus_memory(
        frames()[:-1], visible_until_s=50.0, recent_capacity=2, memory_capacity=2
    )
    aware = timestamp_aware_memory(
        frames()[:-1], visible_until_s=50.0, recent_capacity=2, memory_capacity=2
    )
    assert aware == ordinary
