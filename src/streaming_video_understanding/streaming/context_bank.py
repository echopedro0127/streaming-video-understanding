"""Query-aware causal context probe with bounded ownership of decoded images."""
from __future__ import annotations

from collections import deque
import math

from ..data.schema import FrameRecord


CONTEXT_MODES = ("recent-8", "time-bin-60s-8", "time-bin-prefix-8")


class ContextFrameBank:
    """At most 3*budget retained images (+ incoming frame); no prefix archive.

    Bins use the known query timestamp. This is a diagnostic sampler, not a
    query-agnostic persistent streaming-memory method. Decoder buffers excluded.
    update takes ownership even on errors, except when the image is already owned.
    """

    def __init__(self, *, query_time_s: float, budget: int = 8, horizon_s: float = 60):
        if isinstance(query_time_s, bool) or not math.isfinite(query_time_s) or query_time_s < 0:
            raise ValueError("query time must be finite and non-negative")
        if type(budget) is not int or budget != 8:
            raise ValueError("context probe v1 requires budget=8")
        if isinstance(horizon_s, bool) or not math.isfinite(horizon_s) or horizon_s != 60:
            raise ValueError("context probe v1 requires horizon_s=60")
        self.query_time_s = float(query_time_s)
        self.budget = budget
        self.window_start = max(0.0, self.query_time_s - horizon_s)
        self._recent = deque(maxlen=budget)
        self._window: dict[int, FrameRecord] = {}
        self._prefix: dict[int, FrameRecord] = {}
        self._images = {}
        self._last_id = -1
        self._last_time = -math.inf
        self._closed = False
        self.processed_frames = 0
        self.peak_retained_images = 0

    @property
    def retained_image_count(self) -> int:
        return len(self._images)

    def _consider(self, frame: FrameRecord, start: float, bins: dict[int, FrameRecord]) -> None:
        if frame.timestamp_s < start:
            return
        span = self.query_time_s - start
        index = min(self.budget - 1, int((frame.timestamp_s - start) / span * self.budget)) if span else 0
        centre = start + (index + 0.5) * span / self.budget
        old = bins.get(index)
        def score(item):
            return abs(item.timestamp_s - centre), item.timestamp_s, item.frame_id
        if old is None or score(frame) < score(old):
            bins[index] = frame

    def update(self, image, frame: FrameRecord, *, on_frame=None) -> None:
        already_owned = any(image is owned for owned in self._images.values())
        try:
            if self._closed:
                raise ValueError("context bank is closed")
            if already_owned:
                raise ValueError("image object is already owned by the bank")
            if type(frame.frame_id) is not int or frame.frame_id <= self._last_id:
                raise ValueError("frame IDs must be non-negative and strictly increasing")
            timestamp = frame.timestamp_s
            if isinstance(timestamp, bool) or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("frame timestamp must be finite and non-negative")
            if timestamp < self._last_time:
                raise ValueError("frame timestamps must be chronological")
            if timestamp > self.query_time_s + 1e-6:
                raise ValueError("future frame rejected before storage/rendering")
            if on_frame is not None:
                on_frame(image, frame)
        except BaseException as primary:
            if not already_owned:
                try:
                    image.close()
                except BaseException as cleanup:
                    primary.add_note(f"rejected-frame cleanup failed: {cleanup!r}")
            raise
        self._last_id, self._last_time = frame.frame_id, frame.timestamp_s
        self.processed_frames += 1
        self._images[frame.frame_id] = image
        if frame.timestamp_s >= self.window_start:
            self._recent.append(frame)
        self._consider(frame, self.window_start, self._window)
        self._consider(frame, 0.0, self._prefix)
        retained = {f.frame_id for f in (*self._recent, *self._window.values(), *self._prefix.values())}
        for frame_id in tuple(self._images):
            if frame_id not in retained:
                self._images.pop(frame_id).close()
        self.peak_retained_images = max(self.peak_retained_images, len(self._images))

    def frames(self, mode: str) -> tuple[FrameRecord, ...]:
        if mode == "recent-8":
            values = self._recent
        elif mode == "time-bin-60s-8":
            values = self._window.values()
        elif mode == "time-bin-prefix-8":
            values = self._prefix.values()
        else:
            raise ValueError(f"unknown context mode: {mode}")
        return tuple(sorted(values, key=lambda frame: (frame.timestamp_s, frame.frame_id)))

    def images(self, mode: str) -> tuple:
        return tuple(self._images[frame.frame_id] for frame in self.frames(mode))

    def close(self) -> None:
        errors = []
        for image in self._images.values():
            try:
                image.close()
            except BaseException as exc:
                errors.append(exc)
        self._images.clear()
        self._recent.clear()
        self._window.clear()
        self._prefix.clear()
        self._closed = True
        if errors:
            raise errors[0]
