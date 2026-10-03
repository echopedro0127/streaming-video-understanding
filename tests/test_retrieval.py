from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import threading
import types
import weakref

import numpy as np
import pytest
from PIL import Image

from streaming_video_understanding.data.schema import FrameRecord
from streaming_video_understanding.retrieval import (
    ClipFrameScorer,
    FrameFeatureCache,
    SemanticFrameScorer,
    cosine_similarity_scores,
)


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.moves = []

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self

    def float(self):
        self.value = self.value.astype(np.float32)
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, index):
        return _FakeTensor(self.value[index])

    @property
    def ndim(self):
        return self.value.ndim


class _FakeBatchEncoding(dict):
    pass


class _FakeModelOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


def _install_fake_clip_dependencies(monkeypatch, *, cuda_available=False):
    state = types.SimpleNamespace(
        cuda_available=cuda_available,
        empty_cache_calls=0,
        processor_model_ids=[],
        model_loads=[],
        model_moves=[],
        eval_calls=0,
        text_processor_calls=[],
        image_batches=[],
        text_model_calls=0,
        image_model_calls=0,
        text_feature=np.array([[1.0, 0.0]], dtype=np.float32),
        text_output_style="raw",
        image_output_style="raw",
        image_oom=False,
        text_oom=False,
    )

    class OutOfMemoryError(RuntimeError):
        pass

    torch_module = types.ModuleType("torch")
    torch_module.float16 = object()
    torch_module.float32 = object()
    torch_module.is_tensor = lambda value: isinstance(value, _FakeTensor)
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: state.cuda_available,
        empty_cache=lambda: setattr(
            state, "empty_cache_calls", state.empty_cache_calls + 1
        ),
        OutOfMemoryError=OutOfMemoryError,
    )

    @contextmanager
    def inference_mode():
        yield

    torch_module.inference_mode = inference_mode

    class FakeProcessor:
        def __call__(
            self, *, text=None, images=None, return_tensors=None, padding=None
        ):
            assert return_tensors == "pt"
            if text is not None:
                state.text_processor_calls.append(
                    {"text": text, "padding": padding}
                )
                return _FakeBatchEncoding({
                    "input_ids": _FakeTensor([[1, 2]]),
                    "attention_mask": _FakeTensor([[1, 1]]),
                    "position_ids": _FakeTensor([[0, 1]]),
                    "unexpected_text_key": _FakeTensor([[99]]),
                })
            image_list = list(images)
            state.image_batches.append(
                [image.info["frame_id"] for image in image_list]
            )
            return _FakeBatchEncoding({
                "pixel_values": _FakeTensor(
                    np.stack([image.info["feature"] for image in image_list])
                ),
                "unexpected_image_key": _FakeTensor([[99]]),
            })

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            state.processor_model_ids.append(model_id)
            return FakeProcessor()

    class FakeModel:
        def to(self, device):
            state.model_moves.append(device)
            return self

        def eval(self):
            state.eval_calls += 1
            return self

        def get_text_features(self, **inputs):
            state.text_model_calls += 1
            state.last_text_inputs = inputs
            if state.text_oom:
                raise OutOfMemoryError("fake text OOM")
            return wrap_output(
                state.text_output_style, _FakeTensor(state.text_feature)
            )

        def get_image_features(self, *, pixel_values):
            state.image_model_calls += 1
            state.last_pixel_values = pixel_values
            if state.image_oom:
                raise OutOfMemoryError("fake image OOM")
            return wrap_output(
                state.image_output_style, _FakeTensor(pixel_values.value)
            )

    def wrap_output(style, tensor):
        if style == "raw":
            return tensor
        if style == "pooler":
            return _FakeModelOutput(tensor)
        if style == "tuple":
            return (_FakeTensor(np.zeros((tensor.value.shape[0], 1, 3))), tensor)
        if style == "ambiguous_tuple":
            return (tensor, _FakeTensor(tensor.value + 1.0))
        if style == "malformed":
            return types.SimpleNamespace(last_hidden_state=tensor)
        raise AssertionError(f"unsupported fake output style: {style}")

    class FakeCLIPModel:
        @classmethod
        def from_pretrained(cls, model_id, *, torch_dtype):
            state.model_loads.append((model_id, torch_dtype))
            return FakeModel()

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoProcessor = FakeAutoProcessor
    transformers_module.CLIPModel = FakeCLIPModel
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    return state, torch_module


def _image(frame_id, feature):
    image = Image.new("RGB", (1, 1))
    image.info["frame_id"] = frame_id
    image.info["feature"] = np.asarray(feature, dtype=np.float32)
    return image


def test_semantic_frame_scorer_protocol_has_causal_only_interface():
    parameters = inspect.signature(SemanticFrameScorer.score_frames).parameters

    assert list(parameters) == [
        "self",
        "video_path",
        "frames",
        "images_by_id",
        "query",
    ]


def test_importing_clip_does_not_import_torch_or_transformers():
    script = """
import builtins
import sys

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers'}:
        raise AssertionError(f'eager heavy import: {name}')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import streaming_video_understanding.retrieval.clip
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.fspath(Path.cwd() / "src")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("device", "cuda_available", "resolved_device", "dtype_name"),
    [
        ("auto", False, "cpu", "float32"),
        ("auto", True, "cuda", "float16"),
        ("cpu", True, "cpu", "float32"),
        ("cuda", True, "cuda", "float16"),
    ],
)
def test_clip_constructor_resolves_device_dtype_and_initializes_model(
    tmp_path,
    monkeypatch,
    device,
    cuda_available,
    resolved_device,
    dtype_name,
):
    state, torch_module = _install_fake_clip_dependencies(
        monkeypatch, cuda_available=cuda_available
    )

    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device=device,
        cache_dir=tmp_path / "cache",
        image_batch_size=2,
    )

    assert state.processor_model_ids == ["fake/clip"]
    assert state.model_loads == [
        ("fake/clip", getattr(torch_module, dtype_name))
    ]
    assert state.model_moves == [resolved_device]
    assert state.eval_calls == 1
    assert isinstance(scorer._cache, FrameFeatureCache)


def test_clip_constructor_rejects_unavailable_explicit_cuda(
    tmp_path, monkeypatch
):
    state, _ = _install_fake_clip_dependencies(
        monkeypatch, cuda_available=False
    )

    with pytest.raises(RuntimeError, match="CUDA.*not available"):
        ClipFrameScorer(
            model_id="fake/clip",
            device="cuda",
            cache_dir=tmp_path / "cache",
            image_batch_size=2,
        )

    assert state.processor_model_ids == []
    assert state.model_loads == []


@pytest.mark.parametrize("model_id", ["", "   ", None, 1, False])
def test_clip_constructor_rejects_invalid_model_id(
    tmp_path, monkeypatch, model_id
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)

    with pytest.raises((TypeError, ValueError), match="model_id"):
        ClipFrameScorer(
            model_id=model_id,
            device="cpu",
            cache_dir=tmp_path / "cache",
            image_batch_size=2,
        )

    assert state.processor_model_ids == []


@pytest.mark.parametrize("device", ["", "gpu", None, 1, False])
def test_clip_constructor_rejects_invalid_device(tmp_path, monkeypatch, device):
    state, _ = _install_fake_clip_dependencies(monkeypatch)

    with pytest.raises((TypeError, ValueError), match="device"):
        ClipFrameScorer(
            model_id="fake/clip",
            device=device,
            cache_dir=tmp_path / "cache",
            image_batch_size=2,
        )

    assert state.processor_model_ids == []


@pytest.mark.parametrize("image_batch_size", [0, -1, True, 1.0, "1", None])
def test_clip_constructor_rejects_invalid_batch_size(
    tmp_path, monkeypatch, image_batch_size
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)

    with pytest.raises((TypeError, ValueError), match="image_batch_size"):
        ClipFrameScorer(
            model_id="fake/clip",
            device="cpu",
            cache_dir=tmp_path / "cache",
            image_batch_size=image_batch_size,
        )

    assert state.processor_model_ids == []


def test_clip_scorer_batches_cache_misses_and_reuses_cached_features(
    tmp_path, monkeypatch
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cpu",
        cache_dir=tmp_path / "cache",
        image_batch_size=2,
    )
    frames = tuple(
        FrameRecord(frame_id, index / 2)
        for index, frame_id in enumerate((9, 3, 7, 1, 5))
    )
    features = {
        9: [1.0, 0.0],
        3: [0.0, 1.0],
        7: [-1.0, 0.0],
        1: [1.0, 1.0],
        5: [1.0, -1.0],
    }
    images = {
        frame_id: _image(frame_id, feature)
        for frame_id, feature in features.items()
    }

    first = scorer.score_frames(
        video_path=tmp_path / "video.mp4",
        frames=frames,
        images_by_id=images,
        query="  action  ",
    )
    second = scorer.score_frames(
        video_path=tmp_path / "video.mp4",
        frames=frames,
        images_by_id=images,
        query="action",
    )

    assert list(first) == [9, 3, 7, 1, 5]
    assert first == second
    assert first == pytest.approx(
        {
            9: 1.0,
            3: 0.0,
            7: -1.0,
            1: 2**-0.5,
            5: 2**-0.5,
        }
    )
    assert state.image_batches == [[9, 3], [7, 1], [5]]
    assert state.image_model_calls == 3
    assert state.text_processor_calls == [
        {"text": ["action"], "padding": True},
        {"text": ["action"], "padding": True},
    ]
    assert state.text_model_calls == 2
    assert all(
        tensor.moves == [(("cpu",), {})]
        for tensor in state.last_text_inputs.values()
    )
    assert state.last_pixel_values.moves == [(('cpu',), {})]


def test_clip_scorer_extracts_pooler_outputs_and_filters_processor_keys(
    tmp_path, monkeypatch
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)
    state.text_output_style = "pooler"
    state.image_output_style = "pooler"
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cpu",
        cache_dir=tmp_path / "cache",
        image_batch_size=2,
    )
    frames = (FrameRecord(4, 0.0), FrameRecord(2, 1.0))
    images = {
        4: _image(4, [1.0, 0.0]),
        2: _image(2, [0.0, 1.0]),
    }

    scores = scorer.score_frames(
        video_path=tmp_path / "video.mp4",
        frames=frames,
        images_by_id=images,
        query="query",
    )

    assert scores == {4: 1.0, 2: 0.0}
    assert set(state.last_text_inputs) == {
        "input_ids",
        "attention_mask",
        "position_ids",
    }
    assert state.last_pixel_values.value.shape == (2, 2)


def test_clip_scorer_supports_unambiguous_tuple_outputs(
    tmp_path, monkeypatch
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)
    state.text_output_style = "tuple"
    state.image_output_style = "tuple"
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cpu",
        cache_dir=tmp_path / "cache",
        image_batch_size=1,
    )

    scores = scorer.score_frames(
        video_path=tmp_path / "video.mp4",
        frames=(FrameRecord(5, 0.0),),
        images_by_id={5: _image(5, [1.0, 0.0])},
        query="query",
    )

    assert scores == {5: 1.0}


def test_clip_scorer_cache_hit_uses_pooler_text_without_image_encoding(
    tmp_path, monkeypatch
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)
    state.text_output_style = "pooler"
    state.image_output_style = "malformed"
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cpu",
        cache_dir=tmp_path / "cache",
        image_batch_size=1,
    )
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(6, 0.0)
    scorer._cache.store(video_path, frame, np.array([1.0, 0.0]))

    scores = scorer.score_frames(
        video_path=video_path,
        frames=(frame,),
        images_by_id={6: _image(6, [0.0, 1.0])},
        query="query",
    )

    assert scores == {6: 1.0}
    assert state.image_model_calls == 0
    assert state.text_model_calls == 1


@pytest.mark.parametrize(
    ("stage", "style"),
    [
        ("image", "malformed"),
        ("image", "ambiguous_tuple"),
        ("text", "malformed"),
        ("text", "ambiguous_tuple"),
    ],
)
def test_clip_scorer_rejects_invalid_or_ambiguous_model_outputs(
    tmp_path, monkeypatch, stage, style
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)
    setattr(state, f"{stage}_output_style", style)
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cpu",
        cache_dir=tmp_path / "cache",
        image_batch_size=1,
    )
    video_path = tmp_path / "video.mp4"
    frame = FrameRecord(7, 0.0)
    if stage == "text":
        scorer._cache.store(video_path, frame, np.array([1.0, 0.0]))

    with pytest.raises(RuntimeError, match=rf"CLIP {stage} output"):
        scorer.score_frames(
            video_path=video_path,
            frames=(frame,),
            images_by_id={7: _image(7, [1.0, 0.0])},
            query="query",
        )


def test_clip_scorer_encodes_only_uncached_frames_in_original_order(
    tmp_path, monkeypatch
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cpu",
        cache_dir=tmp_path / "cache",
        image_batch_size=3,
    )
    video_path = tmp_path / "video.mp4"
    frames = (
        FrameRecord(8, 0.0),
        FrameRecord(2, 1.0),
        FrameRecord(6, 2.0),
        FrameRecord(4, 3.0),
    )
    images = {
        frame.frame_id: _image(frame.frame_id, [1.0, frame.frame_id])
        for frame in frames
    }
    scorer._cache.store(video_path, frames[1], np.array([1.0, 2.0]))
    scorer._cache.store(video_path, frames[3], np.array([1.0, 4.0]))

    scores = scorer.score_frames(
        video_path=video_path,
        frames=frames,
        images_by_id=images,
        query="query",
    )

    assert list(scores) == [8, 2, 6, 4]
    assert state.image_batches == [[8, 6]]


@pytest.mark.parametrize("query", ["", "   ", None, 1, False])
def test_clip_scorer_rejects_invalid_query(tmp_path, monkeypatch, query):
    state, _ = _install_fake_clip_dependencies(monkeypatch)
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cpu",
        cache_dir=tmp_path / "cache",
        image_batch_size=2,
    )

    with pytest.raises((TypeError, ValueError), match="query"):
        scorer.score_frames(
            video_path=tmp_path / "video.mp4",
            frames=(FrameRecord(1, 0.0),),
            images_by_id={1: _image(1, [1.0, 0.0])},
            query=query,
        )

    assert state.image_batches == []
    assert state.text_processor_calls == []


def test_clip_scorer_rejects_missing_image_before_model_work(
    tmp_path, monkeypatch
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cpu",
        cache_dir=tmp_path / "cache",
        image_batch_size=2,
    )

    with pytest.raises(ValueError, match="Missing image.*7"):
        scorer.score_frames(
            video_path=tmp_path / "video.mp4",
            frames=(FrameRecord(7, 0.0),),
            images_by_id={},
            query="query",
        )

    assert state.image_batches == []
    assert state.text_processor_calls == []


@pytest.mark.parametrize(
    "frames",
    [
        (FrameRecord(1, 0.0), FrameRecord(1, 1.0)),
        (FrameRecord(True, 0.0),),
        (FrameRecord(1.0, 0.0),),
    ],
)
def test_clip_scorer_rejects_duplicate_or_non_exact_frame_ids(
    tmp_path, monkeypatch, frames
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cpu",
        cache_dir=tmp_path / "cache",
        image_batch_size=2,
    )

    with pytest.raises(ValueError, match="frame_id"):
        scorer.score_frames(
            video_path=tmp_path / "video.mp4",
            frames=frames,
            images_by_id={frame.frame_id: _image(frame.frame_id, [1.0, 0.0]) for frame in frames},
            query="query",
        )

    assert state.image_batches == []
    assert state.text_processor_calls == []


def test_clip_scorer_empty_frames_returns_without_model_work(
    tmp_path, monkeypatch
):
    state, _ = _install_fake_clip_dependencies(monkeypatch)
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cpu",
        cache_dir=tmp_path / "cache",
        image_batch_size=2,
    )

    scores = scorer.score_frames(
        video_path=tmp_path / "video.mp4",
        frames=(),
        images_by_id={},
        query="query",
    )

    assert scores == {}
    assert state.image_batches == []
    assert state.text_processor_calls == []


@pytest.mark.parametrize("oom_stage", ["image", "text"])
def test_clip_scorer_converts_cuda_oom_and_clears_cache(
    tmp_path, monkeypatch, oom_stage
):
    state, _ = _install_fake_clip_dependencies(
        monkeypatch, cuda_available=True
    )
    scorer = ClipFrameScorer(
        model_id="fake/clip",
        device="cuda",
        cache_dir=tmp_path / "cache",
        image_batch_size=2,
    )
    frames = tuple(FrameRecord(frame_id, float(frame_id)) for frame_id in range(3))
    images = {
        frame.frame_id: _image(frame.frame_id, [1.0, frame.frame_id + 1.0])
        for frame in frames
    }
    setattr(state, f"{oom_stage}_oom", True)

    with pytest.raises(RuntimeError, match="3 frames"):
        scorer.score_frames(
            video_path=tmp_path / "video.mp4",
            frames=frames,
            images_by_id=images,
            query="query",
        )

    assert state.empty_cache_calls == 1


def test_cosine_similarity_scores_exact_example():
    scores = cosine_similarity_scores(
        (10, 11, 12),
        np.array([[1, 0], [0, 2], [-1, 0]]),
        np.array([3, 0]),
    )

    assert scores == {10: 1.0, 11: 0.0, 12: -1.0}
    assert all(type(score) is float for score in scores.values())


@pytest.mark.parametrize(
    ("frame_features", "query_feature", "match"),
    [
        (np.array([[1.0, 0.0]]), np.array([0.0, 0.0]), "query.*zero"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([1.0, 0.0]), "frame.*zero"),
    ],
)
def test_cosine_similarity_scores_rejects_zero_norms(
    frame_features, query_feature, match
):
    with pytest.raises(ValueError, match=match):
        cosine_similarity_scores(
            tuple(range(len(frame_features))), frame_features, query_feature
        )


@pytest.mark.parametrize(
    ("frame_ids", "frame_features", "query_feature", "match"),
    [
        ((1,), np.array([1.0, 2.0]), np.array([1.0, 2.0]), "two-dimensional"),
        ((1,), np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]), "one-dimensional"),
        ((1,), np.empty((1, 0)), np.array([]), "positive"),
        ((1,), np.array([[1.0, 2.0]]), np.array([1.0]), "dimension"),
        ((1, 2), np.array([[1.0, 2.0]]), np.array([1.0, 2.0]), "length"),
        ((1, 1), np.array([[1.0], [2.0]]), np.array([1.0]), "unique"),
        ((True,), np.array([[1.0]]), np.array([1.0]), "exact integers"),
        ((1.0,), np.array([[1.0]]), np.array([1.0]), "exact integers"),
        ((1,), np.array([[np.nan]]), np.array([1.0]), "finite"),
        ((1,), np.array([[1.0]]), np.array([np.inf]), "finite"),
        ((1,), np.array([["1"]]), np.array([1.0]), "numeric"),
    ],
)
def test_cosine_similarity_scores_rejects_invalid_inputs(
    frame_ids, frame_features, query_feature, match
):
    with pytest.raises(ValueError, match=match):
        cosine_similarity_scores(frame_ids, frame_features, query_feature)


def test_cosine_similarity_scores_handles_extreme_finite_vectors():
    maximum = np.finfo(np.float64).max

    scores = cosine_similarity_scores(
        (3, 4),
        np.array([[maximum, maximum], [maximum, -maximum]]),
        np.array([maximum, maximum]),
    )

    assert np.isfinite(list(scores.values())).all()
    assert scores[3] == pytest.approx(1.0)
    assert scores[4] == pytest.approx(0.0, abs=1e-15)


def test_cosine_similarity_scores_preserves_frame_id_order():
    scores = cosine_similarity_scores(
        (8, 2, 5),
        np.eye(3),
        np.array([1.0, 1.0, 1.0]),
    )

    assert list(scores) == [8, 2, 5]


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
