"""Causal pixel-change retention; change candidates are not semantic events."""
from __future__ import annotations

from collections import deque
import math

from PIL import Image, ImageChops, ImageStat


class ChangeMemory:
    """Four recent + four historical images, plus incoming and one32x32 thumbnail.

    update takes ownership except for an already-owned image. Parameters are
    fixed for this diagnostic; no answer-dependent thresholds or query scoring.
    """

    threshold = 0.08
    cooldown_s = 2.0

    def __init__(self, *, query_time_s):
        if isinstance(query_time_s, bool) or not math.isfinite(query_time_s) or query_time_s < 0:
            raise ValueError('query time must be finite and non-negative')
        self.query_time_s = query_time_s
        self._recent = deque()
        self._history = []
        self._previous = None
        self._last_id, self._last_time = -1, -math.inf
        self._last_candidate_time = -math.inf
        self._closed = False
        self.change_candidates = 0
        self.peak_retained_images = 0

    def update(self, image, frame):
        owned = any(image is item[0] for item in (*self._recent, *self._history))
        thumb = None
        try:
            if self._closed:
                raise ValueError('change memory is closed')
            if owned:
                raise ValueError('image is already owned')
            if type(frame.frame_id) is not int or frame.frame_id <= self._last_id:
                raise ValueError('frame IDs must be non-negative and increasing')
            t = frame.timestamp_s
            if isinstance(t, bool) or not math.isfinite(t) or t < 0 or t < self._last_time:
                raise ValueError('timestamps must be finite, non-negative and chronological')
            if t > self.query_time_s + 1e-6:
                raise ValueError('future frame rejected')
            thumb = image.resize((32, 32), Image.Resampling.BILINEAR).convert('RGB')
            difference = 0.0
            if self._previous is not None:
                diff = ImageChops.difference(thumb, self._previous)
                try:
                    difference = sum(ImageStat.Stat(diff).mean) / (3 * 255)
                finally:
                    diff.close()
            candidate = difference >= self.threshold and t - self._last_candidate_time >= self.cooldown_s
            if self._previous is not None:
                self._previous.close()
        except BaseException as primary:
            for pending in (thumb, image if not owned else None):
                if pending is None:
                    continue
                try:
                    pending.close()
                except BaseException as cleanup:
                    primary.add_note(f'rejected-image cleanup failed: {cleanup!r}')
            raise
        self._previous = thumb
        self._last_id, self._last_time = frame.frame_id, t
        if candidate:
            self.change_candidates += 1
            self._last_candidate_time = t
        self._recent.append((image, frame, difference if candidate else 0.0, candidate))
        if len(self._recent) > 4:
            self._history.append(self._recent.popleft())
            self._history.sort(key=lambda item: (-item[2], item[1].timestamp_s, item[1].frame_id))
            if len(self._history) > 4:
                self._history.pop()[0].close()
        self.peak_retained_images = max(self.peak_retained_images, len(self._recent) + len(self._history))

    def _ordered(self):
        return sorted((*self._recent, *self._history), key=lambda item: (item[1].timestamp_s, item[1].frame_id))

    def frames(self):
        return tuple(item[1] for item in self._ordered())

    def images(self):
        return tuple(item[0] for item in self._ordered())

    def history_scores(self):
        return [dict(frame_id=item[1].frame_id, timestamp_s=item[1].timestamp_s,
                     score=item[2], change_candidate=item[3])
                for item in sorted(self._history, key=lambda item: item[1].timestamp_s)]

    def close(self):
        errors = []
        for image in [item[0] for item in (*self._recent, *self._history)] + ([self._previous] if self._previous else []):
            try:
                image.close()
            except BaseException as exc:
                errors.append(exc)
        self._recent.clear()
        self._history.clear()
        self._previous = None
        self._closed = True
        if errors:
            raise errors[0]
