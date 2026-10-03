# Query-Aware Semantic Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a causal, fixed-budget `Recent-4 + Q-only Semantic/Temporal-Diversity Memory-4` policy that runs on RTX 3060 without changing existing baselines or results.

**Architecture:** Keep the existing `sampler.py -> engine.py -> run_eval.py` evaluation path. Add a pure semantic/diversity selector, a small protocol for query-to-frame scoring, and a lazily loaded CLIP scorer with a project-local feature cache; the evaluator passes only the question and causal historical frames to that scorer, then records an auditable selection trace. This plan implements semantic memory only; Q+A hypotheses, answer log-probabilities, confidence fusion, and answer/clarify routing will be specified and implemented after this stage produces a validated retrieval baseline.

**Tech Stack:** Python 3.13, PyTorch, Transformers `CLIPModel`/`AutoProcessor`, NumPy, Pillow, PyYAML, pytest, existing SmolVLM backend

---

## Scope and repository map

Run every command from:

```text
E:\NewApplication\CodeToRun\forCV\streaming-video-understanding\.worktrees\optimization-stage1
```

The shared virtual environment is in the main repository, so tests use:

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest
```

Baseline before implementation: `59 passed in 15.68s`.

Files created in this stage:

- `src/streaming_video_understanding/retrieval/__init__.py`: retrieval package exports.
- `src/streaming_video_understanding/retrieval/base.py`: scorer protocol and immutable score result.
- `src/streaming_video_understanding/retrieval/cache.py`: deterministic NumPy feature cache below `.cache/`.
- `src/streaming_video_understanding/retrieval/clip.py`: lazy CLIP image/text encoder and cosine scoring.
- `src/streaming_video_understanding/streaming/semantic.py`: model-independent semantic and temporal-diversity selection.
- `tests/test_semantic_memory.py`: pure selector and no-leakage tests.
- `tests/test_retrieval.py`: cache and cosine-scoring tests without model downloads.
- `configs/semantic-validation-1fps.yaml`: separate semantic experiment config; existing configs stay unchanged.

Files modified in this stage:

- `src/streaming_video_understanding/utils/config.py`: retrieval configuration and validation.
- `src/streaming_video_understanding/evaluation/engine.py`: semantic policy parsing, scoring, selection trace, and retrieval timing.
- `src/streaming_video_understanding/evaluation/metrics.py`: summarize the new `retrieval_ms` timing field without changing old records.
- `scripts/run_eval.py`: create one CLIP scorer only when a selected policy requires it.
- `src/streaming_video_understanding/analysis/summaries.py`: labels/order for new methods while preserving the original four-policy comparison contract.
- `tests/test_config.py`, `tests/test_evaluation.py`, `tests/test_analysis.py`: regression and integration coverage.
- `README.md`, `PROJECT_PLAN.md`: usage, cache size/location, experimental status, and non-claims.

Explicitly out of scope for this plan:

- Q+A option-conditioned retrieval;
- candidate answer token probabilities;
- option-permutation confidence fusion;
- vague-query generation and clarification dialogue;
- changing SmolVLM, training a model, or overwriting any file under `results/`.

### Task 1: Add retrieval configuration without changing existing configs

**Files:**
- Modify: `tests/test_config.py`
- Modify: `src/streaming_video_understanding/utils/config.py`

- [ ] **Step 1: Write failing configuration tests**

Append tests that prove defaults are lightweight and invalid values fail clearly:

```python
def test_load_config_supplies_semantic_retrieval_defaults(tmp_path: Path):
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 8\n  seed: 7\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.retrieval.model_id == "openai/clip-vit-base-patch32"
    assert config.retrieval.device == "auto"
    assert config.retrieval.cache_dir == ".cache/semantic_features"
    assert config.retrieval.image_batch_size == 16
    assert config.retrieval.candidate_multiplier == 4
    assert config.retrieval.cluster_window_s == 3.0
    assert config.retrieval.diversity_weight == 0.25


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_batch_size", 0),
        ("candidate_multiplier", 0),
        ("cluster_window_s", -1),
        ("diversity_weight", -0.1),
        ("diversity_weight", 1.1),
    ],
)
def test_load_config_rejects_invalid_retrieval_values(
    tmp_path: Path, field: str, value: float
):
    config_path = tmp_path / "invalid-retrieval.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 8\n  seed: 7\n"
        f"retrieval:\n  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_config(config_path)
```

- [ ] **Step 2: Run the tests and confirm the missing field failure**

Run:

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_config.py -q
```

Expected: FAIL because `ProjectConfig` has no `retrieval` field.

- [ ] **Step 3: Implement the retrieval dataclass and parser**

Add to `utils/config.py`:

```python
@dataclass(frozen=True)
class RetrievalConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    cache_dir: str = ".cache/semantic_features"
    image_batch_size: int = 16
    candidate_multiplier: int = 4
    cluster_window_s: float = 3.0
    diversity_weight: float = 0.25


@dataclass(frozen=True)
class ProjectConfig:
    dataset: DatasetConfig
    experiment: ExperimentConfig
    model: ModelConfig = ModelConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
```

In `load_config`, parse `raw.get("retrieval", {})`, convert each value to its declared type, and enforce:

```python
if not retrieval_model_id:
    raise ValueError("retrieval.model_id must be non-empty")
if retrieval_device not in {"auto", "cpu", "cuda"}:
    raise ValueError("retrieval.device must be auto, cpu, or cuda")
if not retrieval_cache_dir:
    raise ValueError("retrieval.cache_dir must be non-empty")
if image_batch_size <= 0:
    raise ValueError("retrieval.image_batch_size must be positive")
if candidate_multiplier <= 0:
    raise ValueError("retrieval.candidate_multiplier must be positive")
if not math.isfinite(cluster_window_s) or cluster_window_s <= 0:
    raise ValueError("retrieval.cluster_window_s must be finite and positive")
if not math.isfinite(diversity_weight) or not 0.0 <= diversity_weight <= 1.0:
    raise ValueError("retrieval.diversity_weight must be between 0 and 1")
```

Construct `RetrievalConfig` in the returned `ProjectConfig`. `configuration_id` already serializes dataclass fields through `config.retrieval.__dict__`; add that field to its payload so cache/scoring changes produce a distinct run ID.

- [ ] **Step 4: Run configuration and full regression tests**

Run:

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_config.py -q
& ..\..\.venv\Scripts\python.exe -m pytest -q
```

Expected: configuration tests pass; full suite reports at least `59 passed` with no failures.

- [ ] **Step 5: Commit the configuration increment**

```powershell
git add src/streaming_video_understanding/utils/config.py tests/test_config.py
git commit -m "feat: configure semantic frame retrieval"
```

### Task 2: Implement a pure causal semantic/diversity selector

**Files:**
- Create: `src/streaming_video_understanding/streaming/semantic.py`
- Create: `tests/test_semantic_memory.py`

- [ ] **Step 1: Write selector tests before implementation**

Create tests for relevance, diversity, deterministic ties, missing scores, and future leakage:

```python
import pytest

from streaming_video_understanding.data.schema import FrameRecord
from streaming_video_understanding.streaming.buffer import FutureFrameLeakageError
from streaming_video_understanding.streaming.semantic import select_semantic_history


def _frames() -> tuple[FrameRecord, ...]:
    return tuple(FrameRecord(index, float(index)) for index in range(8))


def test_semantic_history_selects_highest_scores():
    scores = {frame.frame_id: float(frame.frame_id) for frame in _frames()}
    result = select_semantic_history(
        _frames(), scores=scores, visible_until_s=7.0, capacity=3, diversity=False
    )
    assert [frame.frame_id for frame in result.frames] == [5, 6, 7]


def test_diversity_avoids_spending_all_slots_on_adjacent_frames():
    scores = {0: 0.98, 1: 0.10, 2: 0.99, 3: 0.98, 4: 0.97, 5: 0.10, 6: 0.10, 7: 0.98}
    result = select_semantic_history(
        _frames(),
        scores=scores,
        visible_until_s=7.0,
        capacity=3,
        diversity=True,
        candidate_multiplier=3,
        cluster_window_s=2.0,
        diversity_weight=0.5,
    )
    selected = [frame.frame_id for frame in result.frames]
    assert 2 in selected
    assert selected != [2, 3, 4]
    assert len(result.trace) == 3


def test_semantic_history_has_deterministic_tie_breaking():
    result = select_semantic_history(
        _frames(), scores={index: 1.0 for index in range(8)},
        visible_until_s=7.0, capacity=2, diversity=False
    )
    assert [frame.frame_id for frame in result.frames] == [0, 1]


def test_semantic_history_requires_every_candidate_score():
    with pytest.raises(ValueError, match="missing semantic score"):
        select_semantic_history(
            _frames(), scores={0: 1.0}, visible_until_s=7.0, capacity=2
        )


def test_semantic_history_rejects_future_frames():
    with pytest.raises(FutureFrameLeakageError):
        select_semantic_history(
            _frames(), scores={index: 1.0 for index in range(8)},
            visible_until_s=6.0, capacity=2
        )
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_semantic_memory.py -q
```

Expected: collection fails with `ModuleNotFoundError` for `streaming.semantic`.

- [ ] **Step 3: Implement the selector and auditable trace**

Create `streaming/semantic.py` with these public types and signature:

```python
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..data.schema import FrameRecord
from .buffer import FutureFrameLeakageError


@dataclass(frozen=True)
class SemanticSelectionItem:
    frame_id: int
    timestamp_s: float
    semantic_score: float
    objective_score: float


@dataclass(frozen=True)
class SemanticSelection:
    frames: tuple[FrameRecord, ...]
    trace: tuple[SemanticSelectionItem, ...]


def select_semantic_history(
    frames: Sequence[FrameRecord],
    *,
    scores: Mapping[int, float],
    visible_until_s: float,
    capacity: int,
    diversity: bool = False,
    candidate_multiplier: int = 4,
    cluster_window_s: float = 3.0,
    diversity_weight: float = 0.25,
    tolerance_s: float = 1e-6,
) -> SemanticSelection:
```

Implementation rules:

1. Validate non-negative capacity, positive multiplier/window, and diversity weight in `[0, 1]`.
2. Sort frames by `(timestamp_s, frame_id)` and reject any invalid or future timestamp with `FutureFrameLeakageError`.
3. Require one finite score per causal frame.
4. If `diversity=False`, rank by `(-score, timestamp_s, frame_id)`, take `capacity`, and return frames in chronological order.
5. If `diversity=True`, form a candidate pool of `min(len(frames), capacity * candidate_multiplier)` highest-scoring frames. Normalize pool scores to `[0, 1]`; when all scores are equal, use `1.0` for every normalized relevance.
6. Select greedily. For a candidate at time `t`, compute:

```python
redundancy = max(
    (math.exp(-abs(t - chosen.timestamp_s) / cluster_window_s) for chosen in selected),
    default=0.0,
)
objective = normalized_relevance - diversity_weight * redundancy
```

7. At each step choose maximum `(objective, normalized_relevance)` and then the earliest `(timestamp_s, frame_id)` for deterministic ties.
8. Save raw semantic score and the objective used when selected in `SemanticSelectionItem`.

- [ ] **Step 4: Run selector tests**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_semantic_memory.py -q
```

Expected: all selector tests pass.

- [ ] **Step 5: Commit the pure selector**

```powershell
git add src/streaming_video_understanding/streaming/semantic.py tests/test_semantic_memory.py
git commit -m "feat: select causal semantic memory frames"
```

### Task 3: Add a deterministic project-local frame feature cache

**Files:**
- Create: `src/streaming_video_understanding/retrieval/__init__.py`
- Create: `src/streaming_video_understanding/retrieval/cache.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write cache round-trip and key-isolation tests**

```python
from pathlib import Path

import numpy as np

from streaming_video_understanding.data.schema import FrameRecord
from streaming_video_understanding.retrieval.cache import FrameFeatureCache


def test_frame_feature_cache_round_trip(tmp_path: Path):
    cache = FrameFeatureCache(tmp_path, model_id="clip-test")
    frame = FrameRecord(frame_id=7, timestamp_s=3.5)
    expected = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    assert cache.load(video_path=Path("video.mp4"), frame=frame) is None
    cache.store(video_path=Path("video.mp4"), frame=frame, feature=expected)

    np.testing.assert_allclose(
        cache.load(video_path=Path("video.mp4"), frame=frame), expected
    )


def test_frame_feature_cache_separates_models_and_timestamps(tmp_path: Path):
    frame_a = FrameRecord(frame_id=1, timestamp_s=1.0)
    frame_b = FrameRecord(frame_id=1, timestamp_s=2.0)
    first = FrameFeatureCache(tmp_path, model_id="model-a")
    second = FrameFeatureCache(tmp_path, model_id="model-b")
    first.store(Path("video.mp4"), frame_a, np.asarray([1.0], dtype=np.float32))

    assert first.load(Path("video.mp4"), frame_b) is None
    assert second.load(Path("video.mp4"), frame_a) is None
```

- [ ] **Step 2: Run tests and confirm the retrieval package is missing**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -q
```

Expected: collection fails because `retrieval.cache` does not exist.

- [ ] **Step 3: Implement cache keys and atomic NumPy writes**

Implement `FrameFeatureCache` as follows:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..data.schema import FrameRecord


class FrameFeatureCache:
    def __init__(self, root: str | Path, *, model_id: str) -> None:
        self.root = Path(root)
        normalized_model = model_id.strip()
        if not normalized_model:
            raise ValueError("model_id must be non-empty")
        self.model_key = hashlib.sha256(normalized_model.encode("utf-8")).hexdigest()[:16]

    def _path(self, video_path: str | Path, frame: FrameRecord) -> Path:
        resolved = Path(video_path).resolve(strict=False).as_posix()
        video_key = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:20]
        frame_key = f"{frame.frame_id}-{frame.timestamp_s:.6f}.npy"
        return self.root / self.model_key / video_key / frame_key

    def load(self, video_path: str | Path, frame: FrameRecord) -> np.ndarray | None:
        path = self._path(video_path, frame)
        if not path.is_file():
            return None
        with path.open("rb") as handle:
            feature = np.load(handle, allow_pickle=False)
        feature = np.asarray(feature, dtype=np.float32)
        if feature.ndim != 1 or not np.all(np.isfinite(feature)):
            raise ValueError(f"invalid cached feature: {path}")
        return feature

    def store(
        self,
        video_path: str | Path,
        frame: FrameRecord,
        feature: np.ndarray,
    ) -> Path:
        value = np.asarray(feature, dtype=np.float32)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("feature must be a finite one-dimensional array")
        path = self._path(video_path, frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        temporary.replace(path)
        return path
```

Export `FrameFeatureCache` from `retrieval/__init__.py`. The configured root is `.cache/semantic_features`, already excluded by `.gitignore`.

- [ ] **Step 4: Run cache tests and inspect ignored files**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -q
git status --short --ignored | Select-String semantic_features
```

Expected: tests pass; any real semantic cache appears with `!!` and is not staged.

- [ ] **Step 5: Commit the cache**

```powershell
git add src/streaming_video_understanding/retrieval tests/test_retrieval.py
git commit -m "feat: cache semantic frame features"
```

### Task 4: Implement the CLIP frame scorer without a new dependency

**Files:**
- Create: `src/streaming_video_understanding/retrieval/base.py`
- Create: `src/streaming_video_understanding/retrieval/clip.py`
- Modify: `src/streaming_video_understanding/retrieval/__init__.py`
- Modify: `tests/test_retrieval.py`

- [ ] **Step 1: Write pure cosine and protocol-level tests**

Append:

```python
from streaming_video_understanding.retrieval.clip import cosine_similarity_scores


def test_cosine_similarity_scores_normalizes_query_and_frames():
    frame_ids = (10, 11, 12)
    frame_features = np.asarray([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]], dtype=np.float32)
    query_feature = np.asarray([3.0, 0.0], dtype=np.float32)

    scores = cosine_similarity_scores(frame_ids, frame_features, query_feature)

    assert scores == {10: 1.0, 11: 0.0, 12: -1.0}


def test_cosine_similarity_scores_rejects_zero_query():
    with pytest.raises(ValueError, match="zero-norm query"):
        cosine_similarity_scores(
            (1,), np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32)
        )
```

- [ ] **Step 2: Run the focused tests and confirm the function is missing**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -q
```

Expected: import fails for `cosine_similarity_scores`.

- [ ] **Step 3: Define the scorer contract**

Create `retrieval/base.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from PIL import Image

from ..data.schema import FrameRecord


class SemanticFrameScorer(Protocol):
    def score_frames(
        self,
        *,
        video_path: Path,
        frames: Sequence[FrameRecord],
        images_by_id: Mapping[int, Image.Image],
        query: str,
    ) -> dict[int, float]:
        """Return one finite query-to-frame similarity for every supplied frame."""
```

The interface intentionally has no answer, ground-truth label, clue interval, query time, or future-frame input.

- [ ] **Step 4: Implement CLIP encoding and cosine scoring**

Create `retrieval/clip.py` with:

```python
def cosine_similarity_scores(
    frame_ids: Sequence[int],
    frame_features: np.ndarray,
    query_feature: np.ndarray,
) -> dict[int, float]:
    frames = np.asarray(frame_features, dtype=np.float32)
    query = np.asarray(query_feature, dtype=np.float32)
    if frames.ndim != 2 or query.ndim != 1 or frames.shape[1] != query.shape[0]:
        raise ValueError("frame/query feature shapes are incompatible")
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        raise ValueError("zero-norm query feature")
    frame_norms = np.linalg.norm(frames, axis=1)
    if np.any(frame_norms == 0.0):
        raise ValueError("zero-norm frame feature")
    values = (frames / frame_norms[:, None]) @ (query / query_norm)
    return {int(frame_id): float(score) for frame_id, score in zip(frame_ids, values, strict=True)}
```

Implement `ClipFrameScorer` below `cosine_similarity_scores`:

```python
class ClipFrameScorer:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        cache_dir: str | Path,
        image_batch_size: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError(
                "ClipFrameScorer requires torch and transformers"
            ) from exc
        if image_batch_size <= 0:
            raise ValueError("image_batch_size must be positive")
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("retrieval device is cuda but CUDA is unavailable")
        if resolved_device not in {"cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")

        self._torch = torch
        self.device = resolved_device
        self.image_batch_size = image_batch_size
        self.cache = FrameFeatureCache(cache_dir, model_id=model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(resolved_device)
        self.model.eval()

    def _encode_text(self, query: str) -> np.ndarray:
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        moved = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            feature = self.model.get_text_features(**moved)
        return feature[0].detach().float().cpu().numpy()

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with self._torch.inference_mode():
            features = self.model.get_image_features(pixel_values=pixel_values)
        return features.detach().float().cpu().numpy()

    def score_frames(
        self,
        *,
        video_path: Path,
        frames: Sequence[FrameRecord],
        images_by_id: Mapping[int, Image.Image],
        query: str,
    ) -> dict[int, float]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("semantic retrieval query must be non-empty")
        ordered = tuple(frames)
        if not ordered:
            return {}
        if len({frame.frame_id for frame in ordered}) != len(ordered):
            raise ValueError("semantic retrieval requires unique frame IDs")
        missing_images = [frame.frame_id for frame in ordered if frame.frame_id not in images_by_id]
        if missing_images:
            raise ValueError(f"missing decoded images for frame IDs: {missing_images}")

        features: dict[int, np.ndarray] = {}
        misses: list[FrameRecord] = []
        for frame in ordered:
            cached = self.cache.load(video_path, frame)
            if cached is None:
                misses.append(frame)
            else:
                features[frame.frame_id] = cached

        try:
            for start in range(0, len(misses), self.image_batch_size):
                batch = misses[start : start + self.image_batch_size]
                encoded = self._encode_images([images_by_id[frame.frame_id] for frame in batch])
                for frame, feature in zip(batch, encoded, strict=True):
                    self.cache.store(video_path, frame, feature)
                    features[frame.frame_id] = np.asarray(feature, dtype=np.float32)
            query_feature = self._encode_text(normalized_query)
        except self._torch.cuda.OutOfMemoryError as exc:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            raise RuntimeError(
                f"CLIP retrieval ran out of memory for {len(ordered)} frames"
            ) from exc

        matrix = np.stack([features[frame.frame_id] for frame in ordered], axis=0)
        return cosine_similarity_scores(
            tuple(frame.frame_id for frame in ordered), matrix, query_feature
        )
```

Include imports for `Mapping`, `Sequence`, `Path`, `numpy as np`, `PIL.Image`, `FrameRecord`, and `FrameFeatureCache`. Constructor imports `torch`, `AutoProcessor`, and `CLIPModel` only when instantiated, so non-semantic policies do not initialize or download CLIP.

Export `SemanticFrameScorer`, `ClipFrameScorer`, and `cosine_similarity_scores` from `retrieval/__init__.py`.

- [ ] **Step 5: Run retrieval tests without downloading a model**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -q
```

Expected: cache and cosine tests pass; no Hugging Face network request occurs because tests do not instantiate `ClipFrameScorer`.

- [ ] **Step 6: Commit the scorer**

```powershell
git add src/streaming_video_understanding/retrieval tests/test_retrieval.py
git commit -m "feat: score causal frames with CLIP"
```

### Task 5: Integrate semantic policies into the causal evaluator

**Files:**
- Modify: `src/streaming_video_understanding/evaluation/engine.py`
- Modify: `src/streaming_video_understanding/evaluation/metrics.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Add fake-scorer integration tests**

Add a scorer that records exactly what the evaluator sends:

```python
class RecordingScorer:
    def __init__(self, scores: dict[int, float]):
        self.scores = scores
        self.calls: list[dict[str, object]] = []

    def score_frames(self, *, video_path, frames, images_by_id, query):
        self.calls.append({
            "video_path": video_path,
            "frame_ids": tuple(frame.frame_id for frame in frames),
            "image_ids": tuple(sorted(images_by_id)),
            "query": query,
        })
        return {frame.frame_id: self.scores[frame.frame_id] for frame in frames}
```

Add tests asserting:

```python
class TenFrameReader:
    def read_until(self, video_path: Path, end_time_s: float):
        return tuple(
            (Image.new("RGB", (8, 8)), FrameRecord(index, float(index)))
            for index in range(10)
        )


def _semantic_sample(*, answer: str = "A") -> StreamingSample:
    return StreamingSample(
        sample_id="semantic-sample",
        video_id="semantic-video",
        video_path=Path("semantic.mp4"),
        question="Which object appeared earlier?",
        options=("cup", "ball", "book", "phone"),
        answer=answer,
        query_time_s=9.0,
        task_type="Object Recognition",
        temporal_clue_type="Prior",
    )


def _semantic_config() -> ProjectConfig:
    return ProjectConfig(
        dataset=DatasetConfig(context_horizon_s=60.0),
        experiment=ExperimentConfig(visual_budget=4, seed=7),
    )


def test_evaluator_semantic_policy_scores_only_history_and_never_uses_answer():
    sample = _semantic_sample()
    scorer = RecordingScorer({index: float(index) for index in range(8)})
    record = evaluate_sample(
        sample,
        config=_semantic_config(),
        frame_reader=TenFrameReader(),
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        policy_id="recent-2+semantic-memory-2",
        semantic_scorer=scorer,
    )

    assert scorer.calls == [{
        "video_path": sample.video_path,
        "frame_ids": tuple(range(8)),
        "image_ids": tuple(range(10)),
        "query": sample.question,
    }]
    assert "answer" not in scorer.calls[0]
    assert [item["frame_id"] for item in record["selected_memory_frames"]] == [6, 7]
    assert [item["frame_id"] for item in record["selected_recent_frames"]] == [8, 9]
    assert record["input_frames"] == 4
    assert max(record["timestamps_s"]) <= sample.query_time_s


def test_ground_truth_change_does_not_change_semantic_selection():
    scorer = RecordingScorer({index: float(index) for index in range(8)})
    records = [
        evaluate_sample(
            _semantic_sample(answer=answer),
            config=_semantic_config(),
            frame_reader=TenFrameReader(),
            backend=MockVLMBackend(answer="A"),
            output_store=None,
            policy_id="recent-2+semantic-memory-2",
            semantic_scorer=scorer,
        )
        for answer in ("A", "D")
    ]

    assert records[0]["selected_memory_frames"] == records[1]["selected_memory_frames"]
    assert records[0]["semantic_selection_trace"] == records[1]["semantic_selection_trace"]


def test_semantic_policy_requires_a_scorer():
    backend = MockVLMBackend(answer="A")
    with pytest.raises(ValueError, match="requires semantic_scorer"):
        evaluate_sample(
            _semantic_sample(),
            config=_semantic_config(),
            frame_reader=TenFrameReader(),
            backend=backend,
            output_store=None,
            policy_id="recent-2+semantic-memory-2",
        )
    assert backend.call_count == 0
```

Import `Image`, `FrameRecord`, `StreamingSample`, `evaluate_sample`, `MockVLMBackend`, `DatasetConfig`, `ExperimentConfig`, and `ProjectConfig` using the same paths already used elsewhere in this test file.

- [ ] **Step 2: Run the tests and confirm policy parsing fails**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -q
```

Expected: new tests fail with unsupported semantic policy.

- [ ] **Step 3: Extend policy parsing without changing old policy behavior**

Change the combined policy regex in `engine.py` to accept:

```python
r"recent-(\d+)\+(uniform|timestamp-aware|semantic|semantic-diverse)-memory-(\d+)"
```

The two new public policy IDs are:

```text
recent-4+semantic-memory-4
recent-4+semantic-diverse-memory-4
```

Keep `recent-*`, `uniform-*`, and existing timestamp-aware parsing and return values unchanged.

- [ ] **Step 4: Score only causal non-recent history and select memory**

Add to `evaluate_sample`:

```python
semantic_scorer: SemanticFrameScorer | None = None,
```

Before `_select_policy`, parse the policy. For semantic policies:

1. derive recent frames using the existing causal `recent_window`;
2. define history as visible frames not present in recent IDs;
3. reject a missing scorer before calling the answer backend;
4. call `semantic_scorer.score_frames` with `sample.video_path`, history, `images_by_id`, and `sample.question` only;
5. pass the returned mapping into `_select_policy`;
6. use `select_semantic_history` with `diversity=True` only for `semantic-diverse`;
7. keep `include_timestamps=False` for both semantic policies so the first comparison isolates frame selection rather than changing the answer prompt;
8. return and record a `semantic_selection_trace` containing `frame_id`, `timestamp_s`, `semantic_score`, and `objective_score`.

Add `retrieval_latency_ms` to each result record and `retrieval_ms` to stage-level profiles. Old policies record `0.0`. Add `retrieval_ms` to the `stage_fields` tuple in `evaluation/metrics.py` so profile summaries include mean/P50/P95 retrieval time. Keep the existing hard causal assertion after selection.

- [ ] **Step 5: Run evaluation, streaming, and profiling tests**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_evaluation.py tests/test_streaming.py tests/test_profiling.py -q
```

Expected: all focused tests pass, including unchanged recent/uniform/timestamp-aware tests.

- [ ] **Step 6: Commit evaluator integration**

```powershell
git add src/streaming_video_understanding/evaluation/engine.py src/streaming_video_understanding/evaluation/metrics.py tests/test_evaluation.py
git commit -m "feat: evaluate query-aware semantic memory"
```

### Task 6: Load CLIP lazily from the CLI and add a separate experiment config

**Files:**
- Modify: `scripts/run_eval.py`
- Create: `configs/semantic-validation-1fps.yaml`
- Modify: `tests/test_checked_in_configs.py`

- [ ] **Step 1: Add config and lazy-loading tests**

Extend checked-in config tests to load the new file and assert:

```python
config = load_config(Path("configs/semantic-validation-1fps.yaml"))
assert config.experiment.visual_budget == 8
assert config.experiment.policy_ids == (
    "recent-8",
    "uniform-8",
    "recent-4+uniform-memory-4",
    "recent-4+semantic-memory-4",
    "recent-4+semantic-diverse-memory-4",
)
assert config.retrieval.cache_dir == ".cache/semantic_features"
```

Add a small helper test for a new CLI-side function:

```python
assert _requires_semantic_scorer(("recent-8", "uniform-8")) is False
assert _requires_semantic_scorer(("recent-4+semantic-memory-4",)) is True
```

- [ ] **Step 2: Run checked-in config tests and confirm failure**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_checked_in_configs.py -q
```

Expected: FAIL because the semantic config/helper does not exist.

- [ ] **Step 3: Add the semantic validation config**

Create `configs/semantic-validation-1fps.yaml` by copying dataset/model/runtime values from `validation-1fps.yaml`, keeping the 8-frame budget, and replacing policy IDs with the five methods above. Add:

```yaml
retrieval:
  model_id: openai/clip-vit-base-patch32
  device: auto
  cache_dir: .cache/semantic_features
  image_batch_size: 16
  candidate_multiplier: 4
  cluster_window_s: 3.0
  diversity_weight: 0.25
```

Do not edit `configs/validation-1fps.yaml`, `configs/validation-2fps.yaml`, or existing result files.

- [ ] **Step 4: Instantiate the scorer once and only when needed**

In `run_eval.py`, add:

```python
def _requires_semantic_scorer(policy_ids: tuple[str, ...]) -> bool:
    return any("+semantic" in policy_id.lower() for policy_id in policy_ids)
```

After finalizing `policy_ids` and before the sample loop:

```python
semantic_scorer = None
if _requires_semantic_scorer(policy_ids):
    from streaming_video_understanding.retrieval.clip import ClipFrameScorer

    semantic_scorer = ClipFrameScorer(
        model_id=config.retrieval.model_id,
        device=config.retrieval.device,
        cache_dir=config.retrieval.cache_dir,
        image_batch_size=config.retrieval.image_batch_size,
    )
```

Pass `semantic_scorer=semantic_scorer` to every `evaluate_sample` call. This must happen once per process, not once per sample or policy.

- [ ] **Step 5: Run config and synthetic non-semantic smoke tests**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_checked_in_configs.py tests/test_config.py -q
& ..\..\.venv\Scripts\python.exe scripts/run_eval.py --config configs/main.yaml --backend mock --limit 1 --policy recent-8 --output results/semantic-plan-nonsemantic-smoke.jsonl --overwrite
```

Expected: tests pass; smoke prints `processed=1 skipped=0`; no CLIP download occurs for `recent-8`.

- [ ] **Step 6: Commit CLI and experiment config**

```powershell
git add scripts/run_eval.py configs/semantic-validation-1fps.yaml tests/test_checked_in_configs.py
git commit -m "feat: configure semantic memory experiments"
```

### Task 7: Extend reporting labels without breaking the original comparison

**Files:**
- Modify: `src/streaming_video_understanding/analysis/summaries.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write reporting regression tests**

Add tests that assert:

```python
assert POLICY_LABELS["recent-4+semantic-memory-4"] == "Recent-4 + Semantic-Memory-4"
assert POLICY_LABELS["recent-4+semantic-diverse-memory-4"] == (
    "Recent-4 + Semantic-Diverse-Memory-4"
)
```

Create six minimal valid summaries—four original methods and two semantic methods—and assert `ordered_summary_rows` returns the original four first, followed by semantic and semantic-diverse. Retain the existing test proving `build_comparison_rows` still requires exactly the original fixed four methods.

- [ ] **Step 2: Run tests and confirm unknown-policy failure**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_analysis.py -q
```

Expected: new semantic-label test fails.

- [ ] **Step 3: Add extended order while preserving the original contract**

Keep `POLICY_ORDER` unchanged. Add:

```python
SEMANTIC_POLICY_ORDER = (
    "recent-4+semantic-memory-4",
    "recent-4+semantic-diverse-memory-4",
)
DISPLAY_POLICY_ORDER = POLICY_ORDER + SEMANTIC_POLICY_ORDER
```

Add both labels to `POLICY_LABELS`, and make `ordered_summary_rows` use `DISPLAY_POLICY_ORDER`. Keep `_policy_from_summary_path` and `build_comparison_rows` based on `POLICY_ORDER`, so historical four-policy reports remain valid and do not suddenly require semantic results.

- [ ] **Step 4: Run analysis tests**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest tests/test_analysis.py -q
```

Expected: all analysis tests pass, including the old fixed comparison tests.

- [ ] **Step 5: Commit reporting support**

```powershell
git add src/streaming_video_understanding/analysis/summaries.py tests/test_analysis.py
git commit -m "feat: report semantic memory policies"
```

### Task 8: Verify on RTX 3060, document the experiment, and freeze the stage

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_PLAN.md`

- [ ] **Step 1: Run the complete automated test suite**

```powershell
& ..\..\.venv\Scripts\python.exe -m pytest -q
```

Expected: every test passes; test count is greater than 59.

- [ ] **Step 2: Run formatting and repository checks**

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` exits 0; status contains only intended source/test/config/doc changes.

- [ ] **Step 3: Run a real one-sample semantic smoke test**

Set the Hugging Face caches to the E: project and run:

```powershell
$env:HF_HOME = "E:\NewApplication\CodeToRun\forCV\streaming-video-understanding\hf_cache"
$env:HF_HUB_CACHE = "$env:HF_HOME\hub"
& ..\..\.venv\Scripts\python.exe scripts/run_eval.py `
  --config configs/semantic-validation-1fps.yaml `
  --data-root E:\NewApplication\CodeToRun\forCV\streaming-video-understanding\data\streamingbench\real-time-1-100 `
  --annotations annotations.jsonl `
  --backend mock `
  --limit 1 `
  --policy recent-4+semantic-diverse-memory-4 `
  --output results/semantic-memory-smoke.jsonl `
  --overwrite
```

Expected: CLIP downloads below the E: cache on first use, the command prints `processed=1 skipped=0`, output has exactly 4 recent + memory frames in total when enough history exists, every timestamp is at or before the query timestamp, and `semantic_selection_trace` is non-empty. `openai/clip-vit-base-patch32` is well below the project's 10GB pre-download warning threshold; record actual cache growth with `Get-ChildItem` rather than claiming an estimate as a measured value.

- [ ] **Step 4: Run one real SmolVLM answer smoke test**

Repeat Step 3 with `--backend smolvlm` and a different output path:

```powershell
& ..\..\.venv\Scripts\python.exe scripts/run_eval.py `
  --config configs/semantic-validation-1fps.yaml `
  --data-root E:\NewApplication\CodeToRun\forCV\streaming-video-understanding\data\streamingbench\real-time-1-100 `
  --annotations annotations.jsonl `
  --backend smolvlm `
  --limit 1 `
  --policy recent-4+semantic-diverse-memory-4 `
  --output results/semantic-memory-smolvlm-smoke.jsonl `
  --overwrite `
  --profile-output results/semantic-memory-smolvlm-smoke-profile.jsonl
```

Expected: one completed result, no OOM, `input_frames <= 8`, causal timestamps only, and profile includes finite `retrieval_ms` and `total_ms`. If CUDA OOM occurs, first set retrieval `device: cpu` in a temporary untracked config and rerun; do not reduce the 8-frame answer budget or alter existing baseline configs.

- [ ] **Step 5: Document only verified behavior**

Update README with:

- the two new policy IDs and their fixed 4+4 budget;
- the Q-only scorer contract and explicit absence of GT/options in Stage 1 retrieval;
- cache path and first-run model download behavior;
- smoke/full validation commands;
- a statement that semantic retrieval is an experiment, not a proven accuracy improvement until the controlled run completes.

Update `PROJECT_PLAN.md` with completed implementation/test/smoke status and the exact measured test count, model-cache disk growth, peak VRAM, and one-sample latency. Leave full accuracy fields unfilled until the 100-QA controlled experiment actually finishes; phrase them as pending measurements without inserting invented numbers.

- [ ] **Step 6: Commit documentation and verification evidence**

```powershell
git add README.md PROJECT_PLAN.md
git commit -m "docs: document semantic memory evaluation"
```

- [ ] **Step 7: Inspect final branch state**

```powershell
git status --short --branch
git log --oneline -8
```

Expected: clean `feature/optimization-stage1` worktree with small, reviewable commits for configuration, selector, cache, scorer, evaluator, CLI config, reporting, and documentation.

## Stage acceptance checklist

- [ ] Existing four policies produce unchanged selected frame IDs in regression tests.
- [ ] New policies never score recent frames as history and never receive GT answers.
- [ ] Every selected frame satisfies `timestamp_s <= query_time_s + tolerance_s`.
- [ ] Total visual input remains at most 8 frames and uses 4 recent + 4 memory when available.
- [ ] Semantic and semantic-diverse policies differ only in diversity selection.
- [ ] CLIP is loaded once only when a semantic policy is requested.
- [ ] Frame features are cached under ignored project-local `.cache/semantic_features`.
- [ ] Result records contain semantic score/objective traces and retrieval timing.
- [ ] Full test suite passes and one real SmolVLM sample runs without OOM.
- [ ] Existing configs, baselines, result files, and four-policy reporting remain intact.

## Next plan after acceptance

After this stage has controlled retrieval results, create a separate implementation plan for:

1. per-option Q+A hypothesis retrieval as a validation-only ablation;
2. A/B/C/D candidate log-probabilities;
3. retrieval margin/entropy and temporal-cluster ambiguity;
4. option-permutation stability;
5. calibrated answer/abstain routing and risk-coverage metrics;
6. targeted clarification for genuinely ambiguous multi-moment queries.

This boundary keeps retrieval quality and confidence calibration separately measurable.
