import pytest

from streaming_video_understanding.data.video import CachedFrameReader, validate_frame_timestamps


def test_validate_frame_timestamps_rejects_future_frame():
    with pytest.raises(ValueError, match="future"):
        validate_frame_timestamps((1.0, 19.9, 20.1), query_time_s=20.0)


def test_validate_frame_timestamps_accepts_boundary():
    validate_frame_timestamps((1.0, 19.9, 20.0), query_time_s=20.0)


def test_validate_frame_timestamps_rejects_non_chronological_input():
    with pytest.raises(ValueError, match="chronological"):
        validate_frame_timestamps((1.0, 0.5), query_time_s=20.0)


def test_cached_frame_reader_reuses_same_video_query():
    from PIL import Image

    class CountingReader:
        def __init__(self):
            self.calls = 0

        def read_until(self, video_path, end_time_s):
            self.calls += 1
            return iter(
                [
                    (Image.new("RGB", (2, 2)), frame)
                    for frame in range(int(end_time_s) + 1)
                ]
            )

    reader = CountingReader()
    cached = CachedFrameReader(reader, max_cache_frames=8)

    first = list(cached.read_until("video.mp4", 3.0))
    second = list(cached.read_until("video.mp4", 3.0))

    assert reader.calls == 1
    assert len(first) == len(second) == 4


def test_cached_frame_reader_does_not_reuse_different_query_boundary():
    class CountingReader:
        def __init__(self):
            self.calls = 0

        def read_until(self, video_path, end_time_s):
            self.calls += 1
            return iter(())

    reader = CountingReader()
    cached = CachedFrameReader(reader, max_cache_frames=8)

    list(cached.read_until("video.mp4", 3.0))
    list(cached.read_until("video.mp4", 4.0))

    assert reader.calls == 2
