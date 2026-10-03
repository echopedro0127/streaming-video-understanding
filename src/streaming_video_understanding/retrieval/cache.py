from __future__ import annotations

import hashlib
import math
import os
import tempfile
from numbers import Real
from pathlib import Path

import numpy as np

from ..data.schema import FrameRecord


def _digest(value: str, length: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _validated_feature(feature: np.ndarray) -> np.ndarray:
    try:
        converted = np.asarray(feature, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("feature must be a one-dimensional finite numeric array") from exc
    if converted.ndim != 1 or converted.size == 0 or not np.isfinite(converted).all():
        raise ValueError("feature must be a non-empty one-dimensional finite array")
    return converted


class FrameFeatureCache:
    def __init__(self, root: str | Path, *, model_id: str) -> None:
        if not isinstance(model_id, str):
            raise TypeError("model_id must be a non-empty string")
        normalized_model_id = model_id.strip()
        if not normalized_model_id:
            raise ValueError("model_id must be a non-empty string")

        self._root = Path(root)
        self._model_key = _digest(normalized_model_id, 16)

    def _target(self, video_path: str | Path, frame: FrameRecord) -> Path:
        if type(frame.frame_id) is not int:
            raise ValueError(f"frame_id must be an exact integer: {frame.frame_id!r}")
        if (
            not isinstance(frame.timestamp_s, Real)
            or isinstance(frame.timestamp_s, bool)
        ):
            raise ValueError(
                f"frame timestamp must be finite and non-negative: {frame.timestamp_s!r}"
            )
        try:
            timestamp_s = float(frame.timestamp_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"frame timestamp must be finite and non-negative: {frame.timestamp_s!r}"
            ) from exc
        if not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError(
                f"frame timestamp must be finite and non-negative: {frame.timestamp_s!r}"
            )

        resolved_video = Path(video_path).resolve(strict=False).as_posix()
        video_key = _digest(resolved_video, 20)
        filename = f"{frame.frame_id}-{timestamp_s:.6f}.npy"
        return self._root / self._model_key / video_key / filename

    def load(self, video_path: str | Path, frame: FrameRecord) -> np.ndarray | None:
        target = self._target(video_path, frame)
        try:
            with target.open("rb") as handle:
                feature = np.load(handle, allow_pickle=False)
                return _validated_feature(feature)
        except FileNotFoundError:
            return None
        except Exception as exc:
            raise ValueError(f"Invalid cached feature at {target}: {exc}") from exc

    def store(
        self, video_path: str | Path, frame: FrameRecord, feature: np.ndarray
    ) -> Path:
        target = self._target(video_path, frame)
        converted = _validated_feature(feature)
        target.parent.mkdir(parents=True, exist_ok=True)

        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp.npy", dir=target.parent
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            np.save(temp_path, converted, allow_pickle=False)
            temp_path.replace(target)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        return target
