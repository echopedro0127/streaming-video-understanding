import pytest

from streaming_video_understanding.data.video import validate_frame_timestamps


def test_validate_frame_timestamps_rejects_future_frame():
    with pytest.raises(ValueError, match="future"):
        validate_frame_timestamps((1.0, 19.9, 20.1), query_time_s=20.0)


def test_validate_frame_timestamps_accepts_boundary():
    validate_frame_timestamps((1.0, 19.9, 20.0), query_time_s=20.0)


def test_validate_frame_timestamps_rejects_non_chronological_input():
    with pytest.raises(ValueError, match="chronological"):
        validate_frame_timestamps((1.0, 0.5), query_time_s=20.0)
