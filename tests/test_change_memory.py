from PIL import Image
import pytest

from streaming_video_understanding.data.schema import FrameRecord
from streaming_video_understanding.streaming.change_memory import ChangeMemory


def feed(bank, i, value, t=None):
    image = Image.new('RGB', (40, 30), (value, value, value))
    bank.update(image, FrameRecord(i, float(i if t is None else t)))
    return image


def test_changes_retain_early_evidence_with_bounded_ownership():
    bank = ChangeMemory(query_time_s=100)
    images = [feed(bank, i, 255 if 10 <= i < 20 else 0) for i in range(100)]
    assert len(bank.frames()) == 8
    assert [f.frame_id for f in bank.frames()][-4:] == [96, 97, 98, 99]
    assert {10, 20} <= {f.frame_id for f in bank.frames()}
    assert bank.peak_retained_images <= 8
    assert bank.change_candidates == 2
    assert len(bank.history_scores()) == 4
    bank.close()
    for image in images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_cooldown_and_zero_score_fallback():
    bank = ChangeMemory(query_time_s=20)
    for i in range(12):
        feed(bank, i, 255 if i % 2 else 0)
    # Changes at1,3,5,7,9,11; only the first4 evicted candidates survive ties.
    assert bank.change_candidates == 6
    assert [f.frame_id for f in bank.frames()] == [1, 3, 5, 7, 8, 9, 10, 11]
    bank.close()
    flat = ChangeMemory(query_time_s=20)
    for i in range(12):
        feed(flat, i, 0)
    assert [f.frame_id for f in flat.frames()] == [0, 1, 2, 3, 8, 9, 10, 11]
    assert flat.change_candidates == 0
    flat.close()


@pytest.mark.parametrize('timestamp', [21.0, -1.0, float('nan')])
def test_rejected_images_close_before_retention(timestamp):
    bank = ChangeMemory(query_time_s=20)
    image = Image.new('RGB', (8, 8))
    with pytest.raises(ValueError):
        bank.update(image, FrameRecord(0, timestamp))
    assert bank.frames() == ()
    with pytest.raises(ValueError):
        image.getpixel((0, 0))
    bank.close()


def test_reused_owned_image_is_rejected_without_closing_it():
    bank = ChangeMemory(query_time_s=20)
    image = feed(bank, 0, 0)
    with pytest.raises(ValueError, match='owned'):
        bank.update(image, FrameRecord(1, 1.0))
    assert image.getpixel((0, 0)) == (0, 0, 0)
    bank.close()


def test_previous_thumbnail_close_failure_releases_incoming_image(monkeypatch):
    bank = ChangeMemory(query_time_s=20)
    feed(bank, 0, 0)
    image = Image.new('RGB', (8, 8))
    with monkeypatch.context() as patch:
        def fail():
            raise OSError('previous close failed')
        patch.setattr(bank._previous, 'close', fail)
        with pytest.raises(OSError, match='previous close failed'):
            bank.update(image, FrameRecord(1, 1.0))
        with pytest.raises(ValueError):
            image.getpixel((0, 0))
    bank.close()


def test_rejection_cleanup_attempts_image_close_despite_thumbnail_failure(monkeypatch):
    import streaming_video_understanding.streaming.change_memory as module
    bank = ChangeMemory(query_time_s=20)
    feed(bank, 0, 0)
    class Thumb:
        def convert(self, mode):
            return self
        def close(self):
            raise OSError('thumbnail close failed')
    class Incoming:
        closed = False
        def resize(self, *args):
            return Thumb()
        def close(self):
            self.closed = True
    incoming = Incoming()
    def fail(*args):
        raise RuntimeError('difference failed')
    monkeypatch.setattr(module.ImageChops, 'difference', fail)
    with pytest.raises(RuntimeError, match='difference failed') as caught:
        bank.update(incoming, FrameRecord(1, 1.0))
    assert incoming.closed
    assert any('thumbnail close failed' in note for note in caught.value.__notes__)
    bank.close()
