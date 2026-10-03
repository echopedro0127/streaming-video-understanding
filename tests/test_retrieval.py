import hashlib
from pathlib import Path

import numpy as np
import pytest

from streaming_video_understanding.data.schema import FrameRecord
from streaming_video_understanding.retrieval import FrameFeatureCache


def test_missing_store_and_load_round_trip_as_float32(tmp_path):
    cache = FrameFeatureCache(tmp_path / "cache", model_id=" model/example ")
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(frame_id=12, timestamp_s=3.25)

    assert cache.load(video_path, frame) is None

    target = cache.store(video_path, frame, np.array([1.25, -2.5], dtype=np.float64))

    loaded = cache.load(video_path, frame)
    assert target.is_file()
    assert loaded is not None
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, np.array([1.25, -2.5], dtype=np.float32))


def test_same_frame_id_at_different_timestamps_is_isolated(tmp_path):
    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    video_path = tmp_path / "video.mp4"
    first = FrameRecord(frame_id=7, timestamp_s=1.0)
    second = FrameRecord(frame_id=7, timestamp_s=2.0)

    first_path = cache.store(video_path, first, np.array([1.0]))
    second_path = cache.store(video_path, second, np.array([2.0]))

    assert first_path != second_path
    np.testing.assert_array_equal(cache.load(video_path, first), np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(cache.load(video_path, second), np.array([2.0], dtype=np.float32))


def test_same_video_and_frame_under_different_models_is_isolated(tmp_path):
    root = tmp_path / "cache"
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(frame_id=1, timestamp_s=0.5)
    first = FrameFeatureCache(root, model_id="model-a")
    second = FrameFeatureCache(root, model_id="model-b")

    first_path = first.store(video_path, frame, np.array([1.0]))
    second_path = second.store(video_path, frame, np.array([2.0]))

    assert first_path != second_path
    np.testing.assert_array_equal(first.load(video_path, frame), np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(second.load(video_path, frame), np.array([2.0], dtype=np.float32))


def test_different_resolved_video_paths_are_isolated(tmp_path):
    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    frame = FrameRecord(frame_id=1, timestamp_s=0.5)
    first_video = tmp_path / "first" / "video.mp4"
    second_video = tmp_path / "second" / "video.mp4"

    first_path = cache.store(first_video, frame, np.array([1.0]))
    second_path = cache.store(second_video, frame, np.array([2.0]))

    assert first_path != second_path
    np.testing.assert_array_equal(cache.load(first_video, frame), np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(cache.load(second_video, frame), np.array([2.0], dtype=np.float32))


@pytest.mark.parametrize(
    "feature",
    [
        np.array(1.0),
        np.array([[1.0, 2.0]]),
        np.array([], dtype=np.float32),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([-np.inf]),
    ],
)
def test_store_rejects_invalid_features(tmp_path, feature):
    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")

    with pytest.raises(ValueError, match="feature"):
        cache.store(tmp_path / "video.mp4", FrameRecord(0, 0.0), feature)


@pytest.mark.parametrize(
    "cached",
    [
        np.array(1.0),
        np.array([[1.0, 2.0]]),
        np.array([], dtype=np.float32),
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_load_rejects_invalid_cached_array_and_mentions_path(tmp_path, cached):
    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(3, 4.0)
    target = cache.store(video_path, frame, np.array([1.0]))
    with target.open("wb") as handle:
        np.save(handle, cached, allow_pickle=False)

    with pytest.raises(ValueError, match=rf"Invalid cached feature.*{target.name}"):
        cache.load(video_path, frame)


def test_load_rejects_corrupt_file_and_mentions_path(tmp_path):
    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(3, 4.0)
    target = cache.store(video_path, frame, np.array([1.0]))
    target.write_bytes(b"not a numpy file")

    with pytest.raises(ValueError, match=rf"Invalid cached feature.*{target.name}"):
        cache.load(video_path, frame)


def test_path_is_deterministic_normalized_and_leaves_no_temp_files(tmp_path):
    root = tmp_path / "cache"
    video_path = tmp_path / "nested" / ".." / "video.mp4"
    frame = FrameRecord(frame_id=42, timestamp_s=1.2345678)
    cache = FrameFeatureCache(root, model_id="  model/example  ")

    first = cache.store(video_path, frame, np.array([1.0]))
    second = cache.store(video_path, frame, np.array([2.0]))
    expected_model = hashlib.sha256(b"model/example").hexdigest()[:16]
    resolved_video = video_path.resolve(strict=False).as_posix()
    expected_video = hashlib.sha256(resolved_video.encode()).hexdigest()[:20]
    expected = root / expected_model / expected_video / "42-1.234568.npy"

    assert first == second == expected
    assert first.resolve().is_relative_to(root.resolve())
    assert list(first.parent.iterdir()) == [first]
    np.testing.assert_array_equal(cache.load(video_path, frame), np.array([2.0], dtype=np.float32))


def test_store_closes_raw_descriptor_when_fdopen_fails(tmp_path, monkeypatch):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    captured = {}

    def failing_fdopen(descriptor, mode):
        captured["descriptor"] = descriptor
        assert mode == "wb"
        raise OSError("fdopen failure")

    monkeypatch.setattr(cache_module.os, "fdopen", failing_fdopen)

    try:
        with pytest.raises(OSError, match="fdopen failure"):
            cache.store(
                tmp_path / "video.mp4", FrameRecord(0, 0.0), np.array([1.0])
            )

        with pytest.raises(OSError):
            cache_module.os.fstat(captured["descriptor"])
        assert not any((tmp_path / "cache").rglob("*.tmp.npy"))
    finally:
        if "descriptor" in captured:
            try:
                cache_module.os.close(captured["descriptor"])
            except OSError:
                pass
        for temp_path in (tmp_path / "cache").rglob("*.tmp.npy"):
            temp_path.unlink()


def test_store_closes_owned_handle_before_cleaning_save_failure(tmp_path, monkeypatch):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    captured = {}

    def failing_save(handle, feature, *, allow_pickle):
        captured["handle"] = handle
        if handle.closed:
            raise AssertionError("np.save received a closed handle")
        raise OSError("save failure")

    monkeypatch.setattr(cache_module.np, "save", failing_save)

    with pytest.raises(OSError, match="save failure"):
        cache.store(tmp_path / "video.mp4", FrameRecord(0, 0.0), np.array([1.0]))

    assert captured["handle"].closed
    assert not any((tmp_path / "cache").rglob("*.tmp.npy"))


def test_store_closes_handle_before_cleaning_replace_failure(tmp_path, monkeypatch):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    real_save = cache_module.np.save
    captured = {}

    def recording_save(handle, feature, *, allow_pickle):
        captured["handle"] = handle
        real_save(handle, feature, allow_pickle=allow_pickle)

    def failing_replace(path, target):
        captured["closed_at_replace"] = getattr(captured["handle"], "closed", False)
        raise OSError("replace failure")

    monkeypatch.setattr(cache_module.np, "save", recording_save)
    monkeypatch.setattr(cache_module.Path, "replace", failing_replace)

    with pytest.raises(OSError, match="replace failure"):
        cache.store(tmp_path / "video.mp4", FrameRecord(0, 0.0), np.array([1.0]))

    assert captured["closed_at_replace"] is True
    assert captured["handle"].closed
    assert not any((tmp_path / "cache").rglob("*.tmp.npy"))


def test_store_uses_string_cleanup_fallback_when_temp_path_creation_fails(
    tmp_path, monkeypatch
):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    real_mkstemp = cache_module.tempfile.mkstemp
    real_path = cache_module.Path
    captured = {}

    def recording_mkstemp(*args, **kwargs):
        descriptor, temp_name = real_mkstemp(*args, **kwargs)
        captured.update(descriptor=descriptor, temp_name=temp_name)
        return descriptor, temp_name

    def failing_temp_path(value):
        if isinstance(value, str) and value.endswith(".tmp.npy"):
            raise OSError("path construction failure")
        return real_path(value)

    monkeypatch.setattr(cache_module.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(cache_module, "Path", failing_temp_path)

    try:
        with pytest.raises(OSError, match="path construction failure"):
            cache.store(
                tmp_path / "video.mp4", FrameRecord(0, 0.0), np.array([1.0])
            )

        with pytest.raises(OSError):
            cache_module.os.fstat(captured["descriptor"])
        assert not real_path(captured["temp_name"]).exists()
    finally:
        if "descriptor" in captured:
            try:
                cache_module.os.close(captured["descriptor"])
            except OSError:
                pass
        if "temp_name" in captured:
            real_path(captured["temp_name"]).unlink(missing_ok=True)


def test_store_cleanup_failure_does_not_mask_write_failure(tmp_path, monkeypatch):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")

    def failing_save(handle, feature, *, allow_pickle):
        raise OSError("write failure")

    def failing_unlink(path, *, missing_ok):
        raise PermissionError("cleanup failure")

    monkeypatch.setattr(cache_module.np, "save", failing_save)
    monkeypatch.setattr(cache_module.Path, "unlink", failing_unlink)

    with pytest.raises(OSError, match="write failure"):
        cache.store(tmp_path / "video.mp4", FrameRecord(0, 0.0), np.array([1.0]))

    monkeypatch.undo()
    for temp_path in (tmp_path / "cache").rglob("*.tmp.npy"):
        temp_path.unlink()


@pytest.mark.parametrize("frame_id", [True, False, 1.0, "1", None])
def test_rejects_non_exact_integer_frame_ids(tmp_path, frame_id):
    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")

    with pytest.raises(ValueError, match="frame_id"):
        cache.load(tmp_path / "video.mp4", FrameRecord(frame_id, 0.0))


@pytest.mark.parametrize(
    "timestamp",
    [True, "1.0", None, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_rejects_invalid_frame_timestamps(tmp_path, timestamp):
    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")

    with pytest.raises(ValueError, match="timestamp"):
        cache.load(tmp_path / "video.mp4", FrameRecord(0, timestamp))


@pytest.mark.parametrize("model_id", ["", "   ", None, 123, False])
def test_rejects_invalid_model_id(tmp_path, model_id):
    with pytest.raises((TypeError, ValueError), match="model_id"):
        FrameFeatureCache(tmp_path / "cache", model_id=model_id)
