# Streaming Video Understanding MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a causal, resumable StreamingBench evaluator that compares four bounded-frame policies and records accuracy, latency, GPU memory, and failure evidence.

**Architecture:** The evaluator converts each benchmark row into a typed query with a dataset timestamp and 60-second visible context. A causal frame buffer and bounded memory expose only frames at or before that timestamp to deterministic samplers. A backend interface separates policy/evaluator tests from optional SmolVLM inference, while JSONL records make runs resumable and auditable.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Accelerate, Pillow, PyAV, PyYAML, psutil, pytest, FFmpeg/ffprobe, uv for environment management.

---

## File map

- Create `README.md`: public project overview and reproducibility instructions.
- Create `PROJECT_PLAN.md`: phase status and experiment checklist.
- Create `requirements.txt`: direct runtime and test dependencies.
- Create `pyproject.toml`: package metadata, pytest configuration, and src layout.
- Create `configs/main.yaml`: fixed-budget main experiment.
- Create `configs/sensitivity.yaml`: B=4 and B=16 sensitivity experiment.
- Create `src/streaming_video_understanding/data/schema.py`: typed sample/frame records and timestamp parsing.
- Create `src/streaming_video_understanding/data/streamingbench.py`: CSV/JSON/Parquet-compatible benchmark loading and subset validation.
- Create `src/streaming_video_understanding/data/video.py`: metadata probing and timestamp-bounded frame decoding.
- Create `src/streaming_video_understanding/streaming/buffer.py`: causal frame buffer and leakage error.
- Create `src/streaming_video_understanding/streaming/memory.py`: bounded temporal memory state.
- Create `src/streaming_video_understanding/streaming/sampler.py`: four deterministic frame policies.
- Create `src/streaming_video_understanding/models/base.py`: backend protocol and prediction record.
- Create `src/streaming_video_understanding/models/mock.py`: deterministic test backend.
- Create `src/streaming_video_understanding/models/prompt.py`: multiple-choice prompt and answer-letter parser.
- Create `src/streaming_video_understanding/models/smolvlm.py`: optional Transformers backend with clear dependency errors.
- Create `src/streaming_video_understanding/evaluation/resume.py`: completed-key store and atomic JSONL append.
- Create `src/streaming_video_understanding/evaluation/metrics.py`: latency percentiles, accuracy, throughput, and memory summaries.
- Create `src/streaming_video_understanding/evaluation/engine.py`: one-query causal evaluation loop.
- Create `src/streaming_video_understanding/evaluation/failures.py`: failure export and selected-frame manifest.
- Create `src/streaming_video_understanding/utils/config.py`: YAML validation and configuration IDs.
- Create `src/streaming_video_understanding/utils/seed.py`: deterministic seed setup.
- Create `scripts/download_data.py`: explicit, bounded StreamingBench archive download.
- Create `scripts/run_eval.py`: resumable evaluation CLI.
- Create `scripts/benchmark_runtime.py`: aggregate results CLI.
- Create `scripts/render_case.py`: contact-sheet CLI.
- Create tests under `tests/` for every production component.

## Task 1: Create the Python package and configuration contract

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `src/streaming_video_understanding/__init__.py`
- Create: `src/streaming_video_understanding/utils/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing configuration test.**

```python
from pathlib import Path

import pytest

from streaming_video_understanding.utils.config import load_config


def test_load_config_requires_causal_horizon_and_budget(tmp_path: Path):
    config_path = tmp_path / "main.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 8\n  seed: 7\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dataset.context_horizon_s == 60.0
    assert config.experiment.visual_budget == 8
    assert config.experiment.seed == 7


def test_load_config_rejects_zero_visual_budget(tmp_path: Path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 0\n  seed: 7\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="visual_budget"):
        load_config(config_path)
```

- [ ] **Step 2: Run the test and confirm the expected missing-module failure.**

Run: `python -m pytest tests/test_config.py -q`

Expected: FAIL because `streaming_video_understanding.utils.config` does not exist.

- [ ] **Step 3: Add the minimal package contract.**

`requirements.txt` must list `numpy`, `pandas`, `pillow`, `pyyaml`, `psutil`, `pytest`, `torch`, `transformers`, `accelerate`, `safetensors`, and `av`. The file must state that the matching PyTorch CUDA wheel should be installed before the remaining dependencies.

`config.py` must expose immutable dataclasses `DatasetConfig(context_horizon_s: float, timestamp_tolerance_s: float)`, `ExperimentConfig(visual_budget: int, seed: int)`, and `ProjectConfig(dataset: DatasetConfig, experiment: ExperimentConfig)`, load YAML with `yaml.safe_load`, and raise `ValueError` when required keys are absent or the budget/horizon is non-positive.

- [ ] **Step 4: Run the focused test and then the empty test suite.**

Run: `python -m pytest tests/test_config.py -q`

Expected: 2 passed.

Run: `python -m pytest -q`

Expected: the same 2 tests pass.

- [ ] **Step 5: Commit the package contract.**

```text
git add requirements.txt pyproject.toml src tests/test_config.py
git commit -m "feat: add configuration contract"
```

## Task 2: Implement typed samples and StreamingBench parsing

**Files:**
- Create: `src/streaming_video_understanding/data/schema.py`
- Create: `src/streaming_video_understanding/data/streamingbench.py`
- Test: `tests/test_schema.py`
- Test: `tests/test_streamingbench.py`
- Create: `tests/fixtures/streamingbench_sample.csv`

- [ ] **Step 1: Write failing tests for timestamp parsing and row loading.**

```python
from pathlib import Path

import pytest

from streaming_video_understanding.data.schema import parse_timestamp
from streaming_video_understanding.data.streamingbench import load_streamingbench_csv


def test_parse_timestamp_accepts_hh_mm_ss():
    assert parse_timestamp("00:01:22") == 82.0


def test_parse_timestamp_rejects_malformed_value():
    with pytest.raises(ValueError, match="timestamp"):
        parse_timestamp("one minute")


def test_loader_preserves_options_and_query_time(tmp_path: Path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "question_id,task_type,question,time_stamp,answer,options,frames_required,temporal_clue_type,video\n"
        "v1_1,Object Perception,What is visible?,00:00:04,B,\"['A. cat', 'B. dog', 'C. ball', 'D. car']\",single,Prior,data/v1.mp4\n",
        encoding="utf-8",
    )

    rows = load_streamingbench_csv(csv_path)

    assert rows[0].sample_id == "v1_1"
    assert rows[0].query_time_s == 4.0
    assert rows[0].options == ("cat", "dog", "ball", "car")
    assert rows[0].answer == "B"
```

- [ ] **Step 2: Run the tests and confirm they fail because the data modules are missing.**

Run: `python -m pytest tests/test_schema.py tests/test_streamingbench.py -q`

Expected: FAIL with import errors for the missing modules.

- [ ] **Step 3: Implement the minimal data types and parser.**

Define frozen dataclasses:

```python
@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp_s: float

@dataclass(frozen=True)
class StreamingSample:
    sample_id: str
    video_id: str
    video_path: Path
    question: str
    options: Sequence[str]
    answer: str
    query_time_s: float
    task_type: str
    temporal_clue_type: str
```

`parse_timestamp` must accept `SS`, `MM:SS`, and `HH:MM:SS`, including a decimal seconds field, and reject negative or non-finite values. The loader must parse the options column using `ast.literal_eval`, strip the option letters from each choice, normalize the answer to one uppercase letter, and raise a row-specific `ValueError` for missing fields.

- [ ] **Step 4: Run focused tests and the full suite.**

Run: `python -m pytest tests/test_schema.py tests/test_streamingbench.py -q`

Expected: all data parsing tests pass.

Run: `python -m pytest -q`

Expected: all current tests pass.

- [ ] **Step 5: Commit the data contract.**

```text
git add src/streaming_video_understanding/data tests/test_schema.py tests/test_streamingbench.py tests/fixtures
git commit -m "feat: parse StreamingBench samples"
```

## Task 3: Implement causal buffers, bounded memory, and deterministic policies

**Files:**
- Create: `src/streaming_video_understanding/streaming/buffer.py`
- Create: `src/streaming_video_understanding/streaming/memory.py`
- Create: `src/streaming_video_understanding/streaming/sampler.py`
- Test: `tests/test_streaming.py`

- [ ] **Step 1: Write failing causal and policy tests.**

```python
import pytest

from streaming_video_understanding.data.schema import FrameRecord
from streaming_video_understanding.streaming.buffer import (
    CausalFrameBuffer,
    FutureFrameLeakageError,
)
from streaming_video_understanding.streaming.sampler import (
    recent_window,
    uniform_history,
)


def frames():
    return tuple(FrameRecord(i, float(i * 10)) for i in range(7))


def test_buffer_rejects_frame_after_visible_time():
    buffer = CausalFrameBuffer(visible_until_s=30.0)
    with pytest.raises(FutureFrameLeakageError):
        buffer.add(FrameRecord(4, 40.0))


def test_recent_window_returns_latest_visible_frames():
    selected = recent_window(frames(), visible_until_s=50.0, capacity=3)
    assert [frame.frame_id for frame in selected] == [3, 4, 5]


def test_uniform_history_is_deterministic_and_causal():
    selected = uniform_history(frames(), visible_until_s=50.0, capacity=4)
    assert [frame.frame_id for frame in selected] == [0, 1, 3, 5]
    assert max(frame.timestamp_s for frame in selected) <= 50.0
```

- [ ] **Step 2: Run the focused tests and confirm the expected import failure.**

Run: `python -m pytest tests/test_streaming.py -q`

Expected: FAIL because the streaming modules do not exist.

- [ ] **Step 3: Implement causal state and policy functions.**

`CausalFrameBuffer.add` must validate finite timestamps and reject values above `visible_until_s + tolerance`. `snapshot` must return only frames within the configured context interval and in chronological order.

`BoundedTemporalMemory` must expose `update(frame)`, `snapshot()`, and `capacity`, retain at most `capacity` frames, and select one representative per temporal bin when compression is required. It must never accept a future frame.

Implement policy functions with the shared signature:

```python
def select_frames(
    frames: Sequence[FrameRecord],
    *,
    visible_until_s: float,
    capacity: int,
    tolerance_s: float = 1e-6,
) -> Sequence[FrameRecord]: ...
```

Implement `recent_window`, `uniform_history`, `recent_plus_memory`, and `timestamp_aware_memory`. The last two must return the same frame IDs for identical input and budget; timestamp awareness changes prompt metadata, not frame selection.

- [ ] **Step 4: Run streaming tests and verify deterministic behavior.**

Run: `python -m pytest tests/test_streaming.py -q`

Expected: all causal, capacity, ordering, and deterministic-selection tests pass.

Run: `python -m pytest -q`

Expected: all previous tests remain green.

- [ ] **Step 5: Commit the streaming core.**

```text
git add src/streaming_video_understanding/streaming tests/test_streaming.py
git commit -m "feat: add causal bounded temporal policies"
```

## Task 4: Add prompt construction, answer parsing, and model backends

**Files:**
- Create: `src/streaming_video_understanding/models/base.py`
- Create: `src/streaming_video_understanding/models/mock.py`
- Create: `src/streaming_video_understanding/models/prompt.py`
- Create: `src/streaming_video_understanding/models/smolvlm.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for prompt and answer parsing.**

```python
import pytest

from streaming_video_understanding.models.prompt import build_prompt, parse_answer_letter


def test_timestamp_prompt_contains_absolute_and_relative_time():
    prompt = build_prompt(
        question="What is the person holding?",
        options=("a cup", "a ball", "a book", "a phone"),
        timestamps_s=(12.5, 18.0),
        query_time_s=20.0,
        include_timestamps=True,
    )
    assert "absolute_time=12.50s" in prompt
    assert "relative_to_query=-1.50s" in prompt
    assert "answer with only one letter" in prompt.lower()


def test_parse_answer_accepts_only_unambiguous_letter():
    assert parse_answer_letter("B") == "B"
    assert parse_answer_letter("The answer is C.") == "C"
    with pytest.raises(ValueError, match="ambiguous"):
        parse_answer_letter("A or B")
```

- [ ] **Step 2: Run the model tests and confirm the expected import failure.**

Run: `python -m pytest tests/test_models.py -q`

Expected: FAIL because model modules do not exist.

- [ ] **Step 3: Implement the backend protocol and mock backend.**

Define `ModelRequest` with ordered PIL images, timestamps, question, options, query time, and `include_timestamps`. Define `ModelResponse` with raw text, answer letter, generation latency, and optional token counts.

`MockVLMBackend` must return a deterministic configured answer and record the last request for test assertions. It must not inspect future frames or mutate the request.

`build_prompt` must list choices A-D and frame timestamp metadata when requested. `parse_answer_letter` must normalize a standalone A-D or a sentence containing exactly one answer letter and reject zero/multiple candidate letters.

- [ ] **Step 4: Implement the optional SmolVLM backend.**

Import `torch` and `transformers` inside the backend constructor, not at package import time. Load `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` with `device_map="auto"`, `torch_dtype=torch.float16` on CUDA, and deterministic generation (`do_sample=False`). Use multi-image chat content and one request at a time. On missing dependencies, raise `RuntimeError` naming the installation command; on CUDA OOM, clear the cache and raise a message naming the sample/configuration.

- [ ] **Step 5: Run focused and full tests.**

Run: `python -m pytest tests/test_models.py -q`

Expected: all prompt, parser, and mock-backend tests pass without downloading a model.

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the model interface.**

```text
git add src/streaming_video_understanding/models tests/test_models.py
git commit -m "feat: add model backend interface"
```

## Task 5: Implement resumable evaluation and metrics

**Files:**
- Create: `src/streaming_video_understanding/evaluation/resume.py`
- Create: `src/streaming_video_understanding/evaluation/metrics.py`
- Create: `src/streaming_video_understanding/evaluation/failures.py`
- Create: `src/streaming_video_understanding/evaluation/engine.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests for resume, leakage, and metrics.**

```python
import json

import pytest

from streaming_video_understanding.evaluation.resume import ResumeStore
from streaming_video_understanding.evaluation.metrics import summarize_records


def test_resume_store_skips_completed_sample_config(tmp_path):
    path = tmp_path / "results.jsonl"
    store = ResumeStore(path)
    store.append({"sample_id": "s1", "config_id": "recent-8", "correct": True, "latency_ms": 10.0})
    assert store.is_complete("s1", "recent-8")
    assert not store.is_complete("s1", "uniform-8")


def test_metrics_report_accuracy_and_percentiles():
    summary = summarize_records([
        {"correct": True, "latency_ms": 10.0, "video_id": "v1"},
        {"correct": False, "latency_ms": 20.0, "video_id": "v1"},
        {"correct": True, "latency_ms": 30.0, "video_id": "v2"},
    ])
    assert summary["question_accuracy"] == pytest.approx(2 / 3)
    assert summary["video_macro_accuracy"] == pytest.approx(0.75)
    assert summary["p50_latency_ms"] == pytest.approx(20.0)


def test_evaluator_rejects_future_frame_before_backend_call():
    from pathlib import Path

    from PIL import Image

    from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
    from streaming_video_understanding.evaluation.engine import evaluate_sample
    from streaming_video_understanding.models.mock import MockVLMBackend
    from streaming_video_understanding.streaming.buffer import FutureFrameLeakageError
    from streaming_video_understanding.utils.config import (
        DatasetConfig,
        ExperimentConfig,
        ProjectConfig,
    )

    class StaticReader:
        def read_until(self, video_path: Path, end_time_s: float):
            return (
                (Image.new("RGB", (8, 8)), FrameRecord(0, 10.0)),
                (Image.new("RGB", (8, 8)), FrameRecord(1, 25.0)),
            )

    sample = StreamingSample(
        sample_id="s1",
        video_id="v1",
        video_path=Path("video.mp4"),
        question="What is visible?",
        options=("cat", "dog", "ball", "car"),
        answer="A",
        query_time_s=20.0,
        task_type="Object Perception",
        temporal_clue_type="Prior",
    )
    backend = MockVLMBackend(answer="A")

    with pytest.raises(FutureFrameLeakageError):
        evaluate_sample(
            sample,
            config=ProjectConfig(
                dataset=DatasetConfig(context_horizon_s=60.0, timestamp_tolerance_s=1e-6),
                experiment=ExperimentConfig(visual_budget=8, seed=7),
            ),
            frame_reader=StaticReader(),
            backend=backend,
            output_store=None,
        )

    assert backend.call_count == 0
```

- [ ] **Step 2: Run tests and confirm the expected missing-module failure.**

Run: `python -m pytest tests/test_evaluation.py -q`

Expected: FAIL because evaluation modules do not exist.

- [ ] **Step 3: Implement atomic resume storage and metric aggregation.**

`ResumeStore` loads existing JSONL keys into memory, appends each record with a flush, and uses `(sample_id, config_id)` as the completion key. Malformed existing lines must be reported with line number and ignored only when `repair=True` is explicit.

`summarize_records` must return question accuracy, video-macro accuracy, mean/P50/P95 latency, throughput, average input frames, average retained frames, compression ratio, and peak GPU memory when fields exist. Percentiles must be deterministic using linear interpolation.

- [ ] **Step 4: Implement the causal evaluator.**

`evaluate_sample(sample, config, frame_reader, backend, output_store)` must:

1. derive the visible interval from the dataset query timestamp and configured horizon;
2. decode or receive only frames at or before the query time;
3. update the bounded memory chronologically;
4. freeze the memory at the query boundary;
5. select frames with the configured policy;
6. assert the maximum selected timestamp is causal;
7. measure decode and model latency separately;
8. call the backend once;
9. write the complete record;
10. export incorrect or failed samples to the failure store.

The evaluator must never call the backend if the leakage assertion fails.

- [ ] **Step 5: Run focused and full tests.**

Run: `python -m pytest tests/test_evaluation.py -q`

Expected: all resume, causal rejection, failure export, and metric tests pass.

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the evaluation engine.**

```text
git add src/streaming_video_understanding/evaluation tests/test_evaluation.py
git commit -m "feat: add resumable causal evaluator"
```

## Task 6: Add frame decoding, data download, and command-line entry points

**Files:**
- Create: `src/streaming_video_understanding/data/video.py`
- Create: `scripts/download_data.py`
- Create: `scripts/run_eval.py`
- Create: `scripts/benchmark_runtime.py`
- Create: `scripts/render_case.py`
- Create: `tests/test_video_tools.py`

- [ ] **Step 1: Write failing tests for frame-bound validation and CLI parsing.**

```python
from pathlib import Path

import pytest

from streaming_video_understanding.data.video import validate_frame_timestamps


def test_validate_frame_timestamps_rejects_future_frame():
    with pytest.raises(ValueError, match="future"):
        validate_frame_timestamps((1.0, 19.9, 20.1), query_time_s=20.0)


def test_validate_frame_timestamps_accepts_boundary():
    validate_frame_timestamps((1.0, 19.9, 20.0), query_time_s=20.0)
```

- [ ] **Step 2: Run the test and confirm the expected import failure.**

Run: `python -m pytest tests/test_video_tools.py -q`

Expected: FAIL because the video tools module does not exist.

- [ ] **Step 3: Implement minimal frame tools and safe download behavior.**

Use PyAV for sequential decoding when installed and FFmpeg/ffprobe as the documented fallback. `FrameReader.read_until(video_path, end_time_s)` must return chronological PIL images plus `FrameRecord` values and must stop decoding once the next frame timestamp is beyond the requested boundary.

`download_data.py` must require `--subset real-time-1-50`, print the exact 9.08GB archive target and expected extraction directory, refuse a destination with less than 25GB free, and require `--confirm-large-download` before downloading. It must not accept a flag that downloads the complete repository.

`run_eval.py` must accept `--config`, `--data-root`, `--backend mock|smolvlm`, `--limit`, `--output`, `--resume`, and `--overwrite`. `benchmark_runtime.py` must read JSONL and write summary JSON/CSV. `render_case.py` must render selected frames with their absolute and relative timestamps without embedding future frames.

- [ ] **Step 4: Run video-tool tests and CLI help checks.**

Run: `python -m pytest tests/test_video_tools.py -q`

Expected: all timestamp-bound tests pass.

Run: `python scripts/download_data.py --help`

Expected: help lists `real-time-1-50` and `--confirm-large-download`.

Run: `python scripts/run_eval.py --help`

Expected: help lists backend, config, limit, resume, and overwrite options.

- [ ] **Step 5: Commit CLI tools.**

```text
git add src/streaming_video_understanding/data/video.py scripts tests/test_video_tools.py
git commit -m "feat: add bounded video tools and evaluation CLIs"
```

## Task 7: Add configs, README, and repository skeleton assets

**Files:**
- Create: `configs/main.yaml`
- Create: `configs/sensitivity.yaml`
- Create: `README.md`
- Create: `assets/.gitkeep`
- Create: `experiments/.gitkeep`
- Create: `results/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Write configuration validation tests for the checked-in YAML files.**

```python
from pathlib import Path

from streaming_video_understanding.utils.config import load_config


def test_checked_in_configs_are_loadable():
    for path in Path("configs").glob("*.yaml"):
        config = load_config(path)
        assert config.dataset.context_horizon_s == 60.0
        assert config.experiment.visual_budget in {4, 8, 16}
```

- [ ] **Step 2: Run the configuration test and confirm it fails because the YAML files are missing.**

Run: `python -m pytest tests/test_checked_in_configs.py -q`

Expected: FAIL because `configs/main.yaml` and `configs/sensitivity.yaml` do not exist.

- [ ] **Step 3: Add the configs and README without numerical claims.**

`main.yaml` must define the 60-second horizon, 1e-6 timestamp tolerance, seed 7, budget 8, recent/memory split 4/4, the four main policy IDs, and SmolVLM model ID. `sensitivity.yaml` must define budgets 4 and 16 and only the Recent-B and Timestamp-aware-Memory-B policy IDs.

README sections must be: Motivation, Research Question, Streaming Protocol, Method, Baselines, Installation, Dataset, Evaluation, Results placeholder, Efficiency, Failure Analysis, Limitations, and Citation. Results must say “not measured yet” until generated artifacts are copied into the repository.

- [ ] **Step 4: Run configuration and repository checks.**

Run: `python -m pytest tests/test_checked_in_configs.py -q`

Expected: all checked-in configuration tests pass.

Run: `git check-ignore -v data models checkpoints results/*.jsonl .worktrees`

Expected: every local-data/cache/result/worktree path is ignored.

- [ ] **Step 5: Commit public documentation and configs.**

```text
git add README.md configs assets experiments results .gitignore tests/test_checked_in_configs.py
git commit -m "docs: add reproducible MVP configuration"
```

## Task 8: Verify the MVP before any real download

**Files:**
- Modify: `PROJECT_PLAN.md`
- Create: `experiments/smoke_manifest.json`

- [ ] **Step 1: Run the complete unit test suite.**

Run: `python -m pytest -q`

Expected: all tests pass with zero failures and no import warnings.

- [ ] **Step 2: Run the mock evaluator on a synthetic three-frame stream.**

Run: `python scripts/run_eval.py --config configs/main.yaml --backend mock --limit 3 --output results/mock-smoke.jsonl --resume`

Expected: three resumable records, zero future-frame leakage violations, one summary-ready JSONL file, and a second identical command that skips all three completed records.

- [ ] **Step 3: Verify the resume and leakage audit artifacts.**

Run: `python scripts/benchmark_runtime.py --input results/mock-smoke.jsonl --output results/mock-smoke-summary.json`

Expected: summary JSON contains `question_accuracy`, `video_macro_accuracy`, `p50_latency_ms`, `p95_latency_ms`, `throughput_samples_per_s`, and `peak_gpu_memory_mb`.

- [ ] **Step 4: Update `PROJECT_PLAN.md` with evidence.**

Record the exact test count, smoke command, output paths, and measured status. Do not write model accuracy or VRAM claims until a real-model command has produced them.

- [ ] **Step 5: Commit the verified MVP foundation.**

```text
git add PROJECT_PLAN.md experiments/smoke_manifest.json
git commit -m "test: verify mock streaming evaluator"
```

## Plan self-review

- StreamingBench primary data and 60-second context are covered by Tasks 2, 6, and 7.
- Four methods and fixed-budget comparisons are covered by Task 3 and the checked-in configs.
- Future-frame leakage is covered by Tasks 3, 5, 6, and 8.
- SmolVLM is optional and no model is downloaded by the foundation tasks; Task 4 isolates it behind a backend interface.
- Resume, deterministic seeds, metrics, OOM-safe one-at-a-time inference, and failure artifacts are covered by Tasks 4–6.
- README, `.gitignore`, public-data restrictions, and no fabricated results are covered by Task 7.
- No task depends on semantic memory or Cosmos3, so either extension can be omitted without blocking the MVP.
