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


def test_cached_frame_reader_bounds_long_video_cache_and_releases_query_images():
    from streaming_video_understanding.data.schema import FrameRecord

    class TrackingImage:
        nbytes = 4

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class CountingReader:
        def __init__(self):
            self.calls = 0
            self.images = []

        def read_until(self, video_path, end_time_s):
            self.calls += 1
            self.images = [TrackingImage() for _ in range(5)]
            return iter(
                (image, FrameRecord(index, float(index)))
                for index, image in enumerate(self.images)
            )

    reader = CountingReader()
    cached = CachedFrameReader(reader, max_cache_frames=2, max_cache_bytes=100)

    decoded = list(cached.read_until("long-video.mp4", 4.0))
    cached.release_query_images("long-video.mp4", tuple(image for image, _ in decoded))

    assert len(decoded) == 5
    assert cached.cached_video_paths == ()
    assert all(image.closed for image in reader.images)


def test_cached_frame_reader_cache_hit_reuses_open_images_after_release():
    from streaming_video_understanding.data.schema import FrameRecord

    class TrackingImage:
        nbytes = 4

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class CountingReader:
        def __init__(self):
            self.calls = 0
            self.images = []

        def read_until(self, video_path, end_time_s):
            self.calls += 1
            image = TrackingImage()
            self.images.append(image)
            return iter(((image, FrameRecord(0, 0.0)),))

    reader = CountingReader()
    cached = CachedFrameReader(reader, max_cache_frames=2, max_cache_bytes=100)

    first = list(cached.read_until("video.mp4", 0.0))
    cached.release_query_images("video.mp4", tuple(image for image, _ in first))
    second = list(cached.read_until("video.mp4", 0.0))

    assert reader.calls == 1
    assert second[0][0] is first[0][0]
    assert not second[0][0].closed
