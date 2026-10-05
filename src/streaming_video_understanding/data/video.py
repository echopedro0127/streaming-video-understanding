from __future__ import annotations

import math
import json
import gc
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .schema import FrameRecord


def _image_nbytes(image: Any) -> int:
    nbytes = getattr(image, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)

    size = getattr(image, "size", None)
    getbands = getattr(image, "getbands", None)
    if size is None or not callable(getbands):
        return 0
    width, height = size
    return int(width) * int(height) * len(getbands())


def validate_frame_timestamps(
    timestamps_s: Sequence[float],
    *,
    query_time_s: float,
    tolerance_s: float = 1e-6,
) -> None:
    if not math.isfinite(query_time_s) or query_time_s < 0:
        raise ValueError("query_time_s must be a finite non-negative number")
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be a finite non-negative number")
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
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if not math.isfinite(timestamp_tolerance_s) or timestamp_tolerance_s < 0:
            raise ValueError("timestamp_tolerance_s must be finite and non-negative")
        self.sample_fps = float(sample_fps)
        self.timestamp_tolerance_s = timestamp_tolerance_s

    def read_until(self, video_path: str | Path, end_time_s: float) -> Iterator[tuple[Image.Image, FrameRecord]]:
        if not math.isfinite(end_time_s) or end_time_s < 0:
            raise ValueError("end_time_s must be a finite non-negative number")
        try:
            import av
        except ImportError:
            yield from self._read_with_ffmpeg(Path(video_path), end_time_s)
            return

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {path}")
        interval_s = 1.0 / self.sample_fps
        next_sample_s = 0.0
        frame_id = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            previous_timestamp_s = -math.inf
            for decoded_index, frame in enumerate(container.decode(stream)):
                timestamp_s = frame.time
                if timestamp_s is None:
                    rate = float(stream.average_rate) if stream.average_rate else 1.0
                    timestamp_s = decoded_index / rate
                timestamp_s = float(timestamp_s)
                if timestamp_s > end_time_s + self.timestamp_tolerance_s:
                    break
                if not math.isfinite(timestamp_s) or timestamp_s < 0:
                    raise ValueError(f"invalid decoded frame timestamp: {timestamp_s!r}")
                if timestamp_s < previous_timestamp_s - self.timestamp_tolerance_s:
                    raise ValueError("decoded frame timestamps must be chronological")
                previous_timestamp_s = timestamp_s
                if timestamp_s + self.timestamp_tolerance_s < next_sample_s:
                    continue
                image = frame.to_image().convert("RGB")
                record = FrameRecord(frame_id=frame_id, timestamp_s=timestamp_s)
                yield image, record
                frame_id += 1
                next_sample_s = timestamp_s + interval_s

    def _read_with_ffmpeg(
        self, video_path: Path, end_time_s: float
    ) -> Iterator[tuple[Image.Image, FrameRecord]]:
        """Fallback reader using ffprobe metadata and an ffmpeg rawvideo pipe."""
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise RuntimeError(
                "FrameReader requires PyAV or ffmpeg/ffprobe; install PyAV with `python -m pip install av`"
            )
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if not streams or not streams[0].get("width") or not streams[0].get("height"):
            raise RuntimeError(f"ffprobe could not determine video dimensions: {video_path}")
        width = int(streams[0]["width"])
        height = int(streams[0]["height"])
        process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps={self.sample_fps:g}",
                "-t",
                f"{end_time_s + self.timestamp_tolerance_s:g}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        frame_size = width * height * 3
        frame_id = 0
        try:
            while process.stdout is not None:
                raw = process.stdout.read(frame_size)
                if not raw:
                    break
                if len(raw) != frame_size:
                    raise RuntimeError("ffmpeg returned a truncated raw video frame")
                timestamp_s = frame_id / self.sample_fps
                if timestamp_s > end_time_s + self.timestamp_tolerance_s:
                    break
                yield Image.frombytes("RGB", (width, height), raw), FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                )
                frame_id += 1
        finally:
            if process.stdout is not None:
                process.stdout.close()
            return_code = process.wait()
            if return_code != 0:
                stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                raise RuntimeError(f"ffmpeg frame decode failed: {stderr.strip()}")


class CachedFrameReader:
    """Optionally reuse one bounded decoded query across policy evaluations."""

    def __init__(
        self,
        reader: Any,
        *,
        max_cache_frames: int = 128,
        max_cache_bytes: int = 64 * 1024 * 1024,
        timestamp_tolerance_s: float = 1e-6,
    ) -> None:
        if max_cache_frames < 0:
            raise ValueError("max_cache_frames must be non-negative")
        if max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")
        if not math.isfinite(timestamp_tolerance_s) or timestamp_tolerance_s < 0:
            raise ValueError("timestamp_tolerance_s must be finite and non-negative")
        self.reader = reader
        self.max_cache_frames = max_cache_frames
        self.max_cache_bytes = max_cache_bytes
        self.timestamp_tolerance_s = timestamp_tolerance_s
        self._cached_path: Path | None = None
        self._cached_end_time_s: float | None = None
        self._cached_frames: tuple[tuple[Image.Image, FrameRecord], ...] = ()
        self._uncacheable_paths: set[Path] = set()

    def read_until(
        self, video_path: str | Path, end_time_s: float
    ) -> Iterator[tuple[Image.Image, FrameRecord]]:
        path = Path(video_path).resolve()
        if path in self._uncacheable_paths:
            yield from self._read_filtered(path, end_time_s)
            return
        if self.max_cache_frames == 0:
            self._uncacheable_paths.add(path)
            yield from self._read_filtered(path, end_time_s)
            return
        cache_hit = (
            self._cached_path == path
            and self._cached_end_time_s is not None
            and abs(self._cached_end_time_s - end_time_s) <= self.timestamp_tolerance_s
        )
        if cache_hit:
            yield from self._cached_frames
            return

        self._clear_cache()
        frames: list[tuple[Image.Image, FrameRecord]] = []
        cached_bytes = 0
        exceeded = False
        for image, frame in self.reader.read_until(video_path, end_time_s):
            frames.append((image, frame))
            cached_bytes += _image_nbytes(image)
            if len(frames) > self.max_cache_frames or cached_bytes > self.max_cache_bytes:
                exceeded = True
                break
        if exceeded:
            self._close_images(frames)
            self._uncacheable_paths.add(path)
            yield from self._read_filtered(path, end_time_s)
            return

        decoded = tuple(frames)
        self._cached_path = path
        self._cached_end_time_s = float(end_time_s)
        self._cached_frames = decoded
        yield from decoded

    def _read_filtered(
        self, video_path: str | Path, end_time_s: float
    ) -> Iterator[tuple[Image.Image, FrameRecord]]:
        yield from self.reader.read_until(video_path, end_time_s)

    def release_query_images(self, video_path: str | Path, images: Sequence[Any]) -> None:
        """Release per-query images when this video cannot be cached safely."""
        path = Path(video_path).resolve()
        if path in self._uncacheable_paths:
            self._close_images(tuple((image, FrameRecord(-1, 0.0)) for image in images))
            gc.collect()
        self._empty_cuda_cache()

    @staticmethod
    def _close_images(frames: Sequence[tuple[Any, FrameRecord]]) -> None:
        for image, _ in frames:
            close = getattr(image, "close", None)
            if close is not None:
                close()

    def _clear_cache(self) -> None:
        self._close_images(self._cached_frames)
        self._cached_path = None
        self._cached_end_time_s = None
        self._cached_frames = ()

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def cached_video_paths(self) -> tuple[Path, ...]:
        return (self._cached_path,) if self._cached_path is not None else ()
