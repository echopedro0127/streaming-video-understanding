from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import os
from pathlib import Path
import threading
import weakref

import numpy as np
import pytest

from streaming_video_understanding.data.schema import FrameRecord
from streaming_video_understanding.retrieval import FrameFeatureCache


class _CloseFailingHandle:
    def __init__(self, handle, *, fail_before_close):
        self._handle = handle
        self._fail_before_close = fail_before_close

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._fail_before_close:
            raise OSError("wrapper close failure before release")
        self._handle.close()
        raise OSError("wrapper close failure after release")

    def cleanup(self):
        try:
            self._handle.close()
        except OSError:
            pass


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


def test_load_rejects_truncated_npy_and_mentions_path(tmp_path):
    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(3, 4.0)
    target = cache.store(video_path, frame, np.array([1.0]))
    target.write_bytes(b"\x93NUMPY\x01\x00")

    with pytest.raises(ValueError, match=rf"Invalid cached feature.*{target.name}"):
        cache.load(video_path, frame)


def test_load_propagates_permission_error_without_calling_it_invalid(
    tmp_path, monkeypatch
):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(3, 4.0)
    target = cache.store(video_path, frame, np.array([1.0]))
    real_open = cache_module.Path.open

    def permission_denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError("sharing violation")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(cache_module.Path, "open", permission_denied)

    with pytest.raises(PermissionError, match="sharing violation"):
        cache.load(video_path, frame)


def test_load_propagates_transient_read_error_without_calling_it_invalid(
    tmp_path, monkeypatch
):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(3, 4.0)
    cache.store(video_path, frame, np.array([1.0]))

    def failing_load(handle, *, allow_pickle):
        raise OSError("transient read failure")

    monkeypatch.setattr(cache_module.np, "load", failing_load)

    with pytest.raises(OSError, match="transient read failure"):
        cache.load(video_path, frame)


def test_path_is_deterministic_normalized_and_leaves_no_temp_files(tmp_path):
    root = tmp_path / "cache"
    video_path = tmp_path / "nested" / ".." / "video.mp4"
    frame = FrameRecord(frame_id=42, timestamp_s=1.2345678)
    cache = FrameFeatureCache(root, model_id="  model/example  ")

    first = cache.store(video_path, frame, np.array([1.0]))
    second = cache.store(video_path, frame, np.array([2.0]))
    expected_model = hashlib.sha256(b"model/example").hexdigest()[:16]
    resolved_video = os.path.normcase(
        os.fspath(video_path.resolve(strict=False))
    ).replace(os.sep, "/")
    expected_video = hashlib.sha256(resolved_video.encode()).hexdigest()[:20]
    expected = root / expected_model / expected_video / "42-1.234568.npy"

    assert first == second == expected
    assert first.resolve().is_relative_to(root.resolve())
    assert list(first.parent.iterdir()) == [first]
    np.testing.assert_array_equal(
        cache.load(video_path, frame), np.array([1.0], dtype=np.float32)
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_video_path_case_is_normalized_on_windows(tmp_path):
    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    video_path = tmp_path / "MixedCase" / "Video.MP4"
    alternate_case = Path(os.fspath(video_path).swapcase())
    frame = FrameRecord(1, 0.5)

    stored = cache.store(video_path, frame, np.array([1.0]))

    assert cache._target(alternate_case, frame) == stored
    np.testing.assert_array_equal(
        cache.load(alternate_case, frame), np.array([1.0], dtype=np.float32)
    )


@pytest.mark.skipif(
    os.name != "nt", reason="Windows extended paths are platform-specific"
)
def test_windows_drive_aliases_share_video_key_and_target_lock(tmp_path):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    normal_video = tmp_path / "video.mp4"
    extended_video = Path("\\\\?\\" + os.fspath(normal_video))
    frame = FrameRecord(1, 0.5)
    normal_target = cache._target(normal_video, frame)
    extended_target = cache._target(extended_video, frame)

    assert normal_target == extended_target
    normal_lock = cache_module._target_lock(normal_target)
    extended_lock = cache_module._target_lock(
        Path("\\\\?\\" + os.fspath(normal_target))
    )
    assert normal_lock is extended_lock


@pytest.mark.skipif(os.name != "nt", reason="Windows UNC paths are platform-specific")
def test_windows_unc_aliases_share_video_key_and_target_lock(tmp_path):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    normal_video = Path(r"\\server\share\folder\video.mp4")
    extended_video = Path(r"\\?\UNC\server\share\folder\video.mp4")
    frame = FrameRecord(1, 0.5)

    assert cache._target(normal_video, frame) == cache._target(extended_video, frame)
    normal_lock = cache_module._target_lock(
        Path(r"\\server\share\folder\entry.npy")
    )
    extended_lock = cache_module._target_lock(
        Path(r"\\?\UNC\server\share\folder\entry.npy")
    )
    assert normal_lock is extended_lock


def test_target_lock_registry_reclaims_unreferenced_locks(tmp_path):
    import streaming_video_understanding.retrieval.cache as cache_module

    target = tmp_path / "cache" / "entry.npy"
    previous_keys = set(cache_module._TARGET_LOCKS)
    lock = cache_module._target_lock(target)
    new_keys = set(cache_module._TARGET_LOCKS) - previous_keys
    assert len(new_keys) == 1
    key = new_keys.pop()
    lock_reference = weakref.ref(lock)

    del lock
    gc.collect()

    assert lock_reference() is None
    assert key not in cache_module._TARGET_LOCKS


def test_store_closes_raw_descriptor_when_fdopen_fails(tmp_path, monkeypatch):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    captured = {}

    def failing_fdopen(descriptor, mode, *, closefd):
        captured["descriptor"] = descriptor
        assert mode == "wb"
        assert closefd is False
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


def test_store_preserves_save_failure_when_wrapper_close_also_fails(
    tmp_path, monkeypatch
):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    real_fdopen = cache_module.os.fdopen
    captured = {}

    def wrapping_fdopen(descriptor, mode, *, closefd):
        assert closefd is False
        handle = _CloseFailingHandle(
            real_fdopen(descriptor, mode, closefd=closefd),
            fail_before_close=True,
        )
        captured.update(descriptor=descriptor, handle=handle)
        return handle

    def failing_save(handle, feature, *, allow_pickle):
        raise OSError("write failure")

    monkeypatch.setattr(cache_module.os, "fdopen", wrapping_fdopen)
    monkeypatch.setattr(cache_module.np, "save", failing_save)

    try:
        with pytest.raises(OSError, match="write failure"):
            cache.store(
                tmp_path / "video.mp4", FrameRecord(0, 0.0), np.array([1.0])
            )

        with pytest.raises(OSError):
            cache_module.os.fstat(captured["descriptor"])
        assert not any((tmp_path / "cache").rglob("*.tmp.npy"))
    finally:
        if "handle" in captured:
            captured["handle"].cleanup()
        for temp_path in (tmp_path / "cache").rglob("*.tmp.npy"):
            temp_path.unlink()


def test_store_close_failure_before_wrapper_release_closes_raw_descriptor(
    tmp_path, monkeypatch
):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    real_fdopen = cache_module.os.fdopen
    captured = {}

    def wrapping_fdopen(descriptor, mode, *, closefd):
        assert closefd is False
        handle = _CloseFailingHandle(
            real_fdopen(descriptor, mode, closefd=closefd),
            fail_before_close=True,
        )
        captured.update(descriptor=descriptor, handle=handle)
        return handle

    monkeypatch.setattr(cache_module.os, "fdopen", wrapping_fdopen)

    try:
        with pytest.raises(OSError, match="wrapper close failure before release"):
            cache.store(
                tmp_path / "video.mp4", FrameRecord(0, 0.0), np.array([1.0])
            )

        with pytest.raises(OSError):
            cache_module.os.fstat(captured["descriptor"])
        assert not any((tmp_path / "cache").rglob("*.tmp.npy"))
    finally:
        if "handle" in captured:
            captured["handle"].cleanup()
        for temp_path in (tmp_path / "cache").rglob("*.tmp.npy"):
            temp_path.unlink()


def test_store_does_not_reclose_descriptor_reused_after_raw_close(
    tmp_path, monkeypatch
):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    real_close = cache_module._REAL_OS_CLOSE
    captured = {}

    def close_and_reuse(descriptor):
        if "replacement" in captured:
            raise AssertionError("raw descriptor was closed more than once")
        real_close(descriptor)
        replacement = open(tmp_path / "unrelated.bin", "wb", buffering=0)
        assert replacement.fileno() == descriptor
        captured["replacement"] = replacement

    def failing_replace(path, target):
        raise OSError("replace failure")

    monkeypatch.setattr(cache_module, "_REAL_OS_CLOSE", close_and_reuse)
    monkeypatch.setattr(cache_module.Path, "replace", failing_replace)

    try:
        with pytest.raises(OSError, match="replace failure"):
            cache.store(
                tmp_path / "video.mp4", FrameRecord(0, 0.0), np.array([1.0])
            )

        assert captured["replacement"].write(b"still writable") == 14
        assert not any((tmp_path / "cache").rglob("*.tmp.npy"))
    finally:
        if "replacement" in captured:
            captured["replacement"].close()
        for temp_path in (tmp_path / "cache").rglob("*.tmp.npy"):
            temp_path.unlink()


def test_concurrent_same_target_stores_publish_one_valid_feature(tmp_path):
    root = tmp_path / "cache"
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(5, 1.5)
    features = [np.array([float(index), index + 0.5]) for index in range(16)]
    caches = [FrameFeatureCache(root, model_id="model") for _ in features]
    barrier = threading.Barrier(len(features))

    def store(cache, feature):
        barrier.wait()
        return cache.store(video_path, frame, feature)

    with ThreadPoolExecutor(max_workers=len(features)) as executor:
        futures = [
            executor.submit(store, cache, feature)
            for cache, feature in zip(caches, features, strict=True)
        ]
        targets = [future.result() for future in futures]

    loaded = caches[0].load(video_path, frame)
    assert loaded is not None
    assert all(target == targets[0] for target in targets)
    assert any(
        np.array_equal(loaded, feature.astype(np.float32)) for feature in features
    )
    assert not any(root.rglob("*.tmp.npy"))


def test_same_target_load_waits_for_store_publication(tmp_path, monkeypatch):
    import streaming_video_understanding.retrieval.cache as cache_module

    root = tmp_path / "cache"
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(5, 1.5)
    feature = np.array([3.0, 4.0])
    writer = FrameFeatureCache(root, model_id="model")
    reader = FrameFeatureCache(root, model_id="model")
    save_started = threading.Event()
    release_save = threading.Event()
    load_requested_lock = threading.Event()
    real_save = cache_module.np.save
    real_target_lock = cache_module._target_lock

    def blocking_save(handle, value, *, allow_pickle):
        real_save(handle, value, allow_pickle=allow_pickle)
        save_started.set()
        assert release_save.wait(timeout=5)

    def recording_target_lock(target):
        if threading.current_thread().name.startswith("cache-reader"):
            load_requested_lock.set()
        return real_target_lock(target)

    monkeypatch.setattr(cache_module.np, "save", blocking_save)
    monkeypatch.setattr(cache_module, "_target_lock", recording_target_lock)

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="cache-writer"
    ) as writers, ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="cache-reader"
    ) as readers:
        stored = writers.submit(writer.store, video_path, frame, feature)
        assert save_started.wait(timeout=5)
        loaded = readers.submit(reader.load, video_path, frame)
        try:
            assert load_requested_lock.wait(timeout=5)
            assert not loaded.done()
        finally:
            release_save.set()

        assert stored.result(timeout=5).is_file()
        np.testing.assert_array_equal(loaded.result(timeout=5), feature.astype(np.float32))


def test_concurrent_stores_for_unrelated_targets_do_not_block_each_other(
    tmp_path, monkeypatch
):
    import streaming_video_understanding.retrieval.cache as cache_module

    cache = FrameFeatureCache(tmp_path / "cache", model_id="model")
    video_path = tmp_path / "video.mp4"
    first_frame = FrameRecord(1, 1.0)
    second_frame = FrameRecord(2, 2.0)
    first_entered_save = threading.Event()
    release_first_save = threading.Event()
    real_save = cache_module.np.save

    def blocking_save(handle, feature, *, allow_pickle):
        real_save(handle, feature, allow_pickle=allow_pickle)
        if feature[0] == 1.0:
            first_entered_save.set()
            assert release_first_save.wait(timeout=5)

    monkeypatch.setattr(cache_module.np, "save", blocking_save)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            cache.store, video_path, first_frame, np.array([1.0])
        )
        assert first_entered_save.wait(timeout=5)
        second = executor.submit(
            cache.store, video_path, second_frame, np.array([2.0])
        )
        try:
            second_target = second.result(timeout=2)
        finally:
            release_first_save.set()
        first_target = first.result(timeout=5)

    assert first_target.is_file()
    assert second_target.is_file()


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
