from __future__ import annotations

import hashlib
import math
import os
import tempfile
import threading
import weakref
from _thread import LockType
from numbers import Real
from pathlib import Path

import numpy as np

from ..data.schema import FrameRecord


_REAL_OS_CLOSE = os.close
_TARGET_LOCKS_GUARD = threading.Lock()
_TARGET_LOCKS: weakref.WeakValueDictionary[str, LockType] = (
    weakref.WeakValueDictionary()
)


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


def _canonical_filesystem_identity(path: str | Path) -> str:
    value = os.fspath(path)
    if os.name == "nt":
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]

    value = os.fspath(Path(value).resolve(strict=False))
    if os.name == "nt":
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]

    return os.path.normcase(os.path.abspath(value)).replace(os.sep, "/")


def _target_lock(target: Path) -> LockType:
    key = _canonical_filesystem_identity(target)
    with _TARGET_LOCKS_GUARD:
        lock = _TARGET_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _TARGET_LOCKS[key] = lock
        return lock


def _load_cached_feature(target: Path) -> np.ndarray:
    with target.open("rb") as handle:
        try:
            feature = np.load(handle, allow_pickle=False)
        except OSError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid cached feature at {target}: {exc}") from exc
    try:
        return _validated_feature(feature)
    except ValueError as exc:
        raise ValueError(f"Invalid cached feature at {target}: {exc}") from exc


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

        resolved_video = _canonical_filesystem_identity(video_path)
        video_key = _digest(resolved_video, 20)
        filename = f"{frame.frame_id}-{timestamp_s:.6f}.npy"
        return self._root / self._model_key / video_key / filename

    def load(self, video_path: str | Path, frame: FrameRecord) -> np.ndarray | None:
        target = self._target(video_path, frame)
        with _target_lock(target):
            try:
                return _load_cached_feature(target)
            except FileNotFoundError:
                return None

    def store(
        self, video_path: str | Path, frame: FrameRecord, feature: np.ndarray
    ) -> Path:
        target = self._target(video_path, frame)
        converted = _validated_feature(feature)
        with _target_lock(target):
            try:
                _load_cached_feature(target)
            except FileNotFoundError:
                pass
            except ValueError:
                pass
            else:
                return target

            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp.npy", dir=target.parent
            )
            try:
                temp_path: Path | None = None
                wrapper = None
                descriptor_owned = True

                temp_path = Path(temp_name)
                wrapper = os.fdopen(descriptor, "wb", closefd=False)
                try:
                    np.save(wrapper, converted, allow_pickle=False)
                    wrapper.flush()
                except BaseException:
                    open_wrapper = wrapper
                    wrapper = None
                    try:
                        open_wrapper.close()
                    except BaseException:
                        pass
                    raise
                open_wrapper = wrapper
                wrapper = None
                open_wrapper.close()
                try:
                    _REAL_OS_CLOSE(descriptor)
                finally:
                    descriptor_owned = False
                temp_path.replace(target)
            except BaseException:
                if wrapper is not None:
                    open_wrapper = wrapper
                    wrapper = None
                    try:
                        open_wrapper.close()
                    except BaseException:
                        pass
                if descriptor_owned:
                    try:
                        _REAL_OS_CLOSE(descriptor)
                    except BaseException:
                        pass
                    descriptor_owned = False
                try:
                    if temp_path is None:
                        os.unlink(temp_name)
                    else:
                        temp_path.unlink(missing_ok=True)
                except BaseException:
                    pass
                raise
        return target
