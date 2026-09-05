from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from pathlib import Path

from PIL import Image

from .schema import FrameRecord


def validate_frame_timestamps(
    timestamps_s: Sequence[float],
    *,
    query_time_s: float,
    tolerance_s: float = 1e-6,
) -> None:
    previous = -math.inf
    for timestamp_s in timestamps_s:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"invalid frame timestamp: {timestamp_s!r}")
        if timestamp < previous - tolerance_s:
            raise ValueError("frame timestamps must be chronological")
        if timestamp > query_time_s + tolerance_s:
            raise ValueError(
                f"future frame at {timestamp:.6f}s exceeds query time {query_time_s:.6f}s"
            )
        previous = timestamp


class FrameReader:
    """Sequential PyAV reader that stops at a causal query boundary."""

    def __init__(self, *, sample_fps: float = 1.0, timestamp_tolerance_s: float = 1e-6) -> None:
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        self.sample_fps = float(sample_fps)
        self.timestamp_tolerance_s = timestamp_tolerance_s

    def read_until(self, video_path: str | Path, end_time_s: float) -> Iterator[tuple[Image.Image, FrameRecord]]:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "FrameReader requires PyAV; install it with `python -m pip install av`"
            ) from exc

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {path}")
        interval_s = 1.0 / self.sample_fps
        next_sample_s = 0.0
        frame_id = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for decoded_index, frame in enumerate(container.decode(stream)):
                timestamp_s = frame.time
                if timestamp_s is None:
                    rate = float(stream.average_rate) if stream.average_rate else 1.0
                    timestamp_s = decoded_index / rate
                timestamp_s = float(timestamp_s)
                if timestamp_s > end_time_s + self.timestamp_tolerance_s:
                    break
                if timestamp_s + self.timestamp_tolerance_s < next_sample_s:
                    continue
                image = frame.to_image().convert("RGB")
                record = FrameRecord(frame_id=frame_id, timestamp_s=timestamp_s)
                yield image, record
                frame_id += 1
                next_sample_s = timestamp_s + interval_s
