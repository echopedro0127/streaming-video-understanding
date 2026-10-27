"""Posthoc artifact control; no model inference or latency estimates."""
from PIL import Image

from ..streaming.event_state import EventCounter


def is_exact_black(image: Image.Image) -> bool:
    """Reject only all-zero decoded RGB, including a check of every channel."""
    with image.convert('RGB') as rgb:
        return all(extrema == (0, 0) for extrema in rgb.getextrema())


def replay_gated(rows, black_flags):
    """Causal raw1/1 replay: black is unknown, never an event boundary."""
    if len(rows) != len(black_flags) or any(type(flag) is not bool for flag in black_flags):
        raise ValueError('exactly one flag (boolean) required per source row')
    counter = EventCounter(enter_frames=1, exit_frames=1)
    traces = []
    for row, black in zip(rows, black_flags):
        trace = counter.update(None if black else row['category'], row['timestamp_s'])
        traces.append(dict(trace, state=counter.snapshot()))
    return traces
