import pytest

from streaming_video_understanding.data.schema import FrameRecord
from streaming_video_understanding.streaming.context_bank import ContextFrameBank


class Image:
    def __init__(self):
        self.closed = 0
    def close(self):
        self.closed += 1


def test_streaming_bank_has_bounded_image_storage_and_closes_evicted_frames():
    images = [Image() for _ in range(501)]
    bank = ContextFrameBank(query_time_s=500, budget=8, horizon_s=60)
    for index, image in enumerate(images):
        bank.update(image, FrameRecord(index, float(index)))
        assert bank.retained_image_count <= 24
        assert sum(image.closed == 0 for image in images[:index+1]) <= 24
    assert [f.frame_id for f in bank.frames("recent-8")] == list(range(493, 501))
    assert len(bank.frames("time-bin-prefix-8")) == 8
    assert min(f.timestamp_s for f in bank.frames("time-bin-60s-8")) >= 440
    assert max(f.timestamp_s for f in bank.frames("time-bin-prefix-8")) <= 500
    assert bank.peak_retained_images <= 24
    assert bank.images("recent-8")[0] is images[493]
    bank.close()
    assert all(image.closed == 1 for image in images)
    assert bank.retained_image_count == 0


def test_bin_centres_tie_to_earlier_frame_and_short_prefix_modes_match():
    bank = ContextFrameBank(query_time_s=16, budget=8, horizon_s=60)
    for index in range(17):
        bank.update(Image(), FrameRecord(index, float(index)))
    assert [f.frame_id for f in bank.frames("time-bin-prefix-8")] == [1, 3, 5, 7, 9, 11, 13, 15]
    assert bank.frames("time-bin-prefix-8") == bank.frames("time-bin-60s-8")
    bank.close()
    bank = ContextFrameBank(query_time_s=32, budget=8, horizon_s=60)
    for index in range(1, 32, 2):
        bank.update(Image(), FrameRecord(index, float(index)))
    assert [f.frame_id for f in bank.frames("time-bin-prefix-8")] == [1, 5, 9, 13, 17, 21, 25, 29]
    bank.close()


@pytest.mark.parametrize("frame", [FrameRecord(1, 6.0), FrameRecord(1, -1),
                                   FrameRecord(1, float("nan")), FrameRecord(True, 1.0)])
def test_invalid_frame_is_closed_and_not_rendered(frame):
    bank = ContextFrameBank(query_time_s=5)
    image = Image()
    calls = []
    with pytest.raises(ValueError):
        bank.update(image, frame, on_frame=lambda *args: calls.append(True))
    assert image.closed == 1
    assert not calls
    bank.close()


def test_backward_frame_and_callback_errors_do_not_leak_images():
    bank = ContextFrameBank(query_time_s=10)
    first, bad, callback_bad = Image(), Image(), Image()
    bank.update(first, FrameRecord(2, 2.0))
    with pytest.raises(ValueError):
        bank.update(bad, FrameRecord(1, 1.0))
    def fail(*args):
        raise RuntimeError("render failure")
    with pytest.raises(RuntimeError, match="render failure"):
        bank.update(callback_bad, FrameRecord(3, 3.0), on_frame=fail)
    bank.close()
    assert first.closed == bad.closed == callback_bad.closed == 1


def test_recent_respects_empty_last_sixty_seconds_and_input_ownership():
    bank = ContextFrameBank(query_time_s=500)
    image = Image()
    bank.update(image, FrameRecord(0, 0.0))
    assert not bank.frames("recent-8")
    with pytest.raises(ValueError):
        bank.update(image, FrameRecord(1, 1.0))
    assert image.closed == 0
    bank.close()
    assert image.closed == 1


def test_rejected_frame_cleanup_does_not_mask_causal_error():
    class Broken(Image):
        def close(self):
            super().close()
            raise RuntimeError("close failed")
    bank = ContextFrameBank(query_time_s=5)
    image = Broken()
    with pytest.raises(ValueError, match="future") as error:
        bank.update(image, FrameRecord(0, 6.0))
    assert image.closed == 1
    assert any("close failed" in note for note in error.value.__notes__)
    bank.close()
