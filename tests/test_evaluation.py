import json
from pathlib import Path

import pytest

from streaming_video_understanding.evaluation.metrics import summarize_records
from streaming_video_understanding.evaluation.resume import ResumeStore


def test_resume_store_skips_completed_sample_config(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    store = ResumeStore(path)
    store.append({"sample_id": "s1", "config_id": "recent-8", "correct": True})

    assert store.is_complete("s1", "recent-8")
    assert not store.is_complete("s1", "uniform-8")
    assert json.loads(path.read_text(encoding="utf-8"))["sample_id"] == "s1"


def test_resume_store_reports_malformed_lines_unless_repair_is_explicit(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        ResumeStore(path)

    repaired = ResumeStore(path, repair=True)
    assert repaired.completed_keys == set()


def test_metrics_report_accuracy_and_percentiles():
    summary = summarize_records(
        [
            {"correct": True, "latency_ms": 10.0, "video_id": "v1"},
            {"correct": False, "latency_ms": 20.0, "video_id": "v1"},
            {"correct": True, "latency_ms": 30.0, "video_id": "v2"},
        ]
    )

    assert summary["question_accuracy"] == pytest.approx(2 / 3)
    assert summary["video_macro_accuracy"] == pytest.approx(0.75)
    assert summary["p50_latency_ms"] == pytest.approx(20.0)
    assert summary["p95_latency_ms"] == pytest.approx(29.0)


def test_metrics_report_option_position_distributions():
    summary = summarize_records(
        [
            {"correct": True, "latency_ms": 10.0, "video_id": "v1", "ground_truth": "A", "prediction": "A"},
            {"correct": False, "latency_ms": 20.0, "video_id": "v1", "ground_truth": "B", "prediction": "C"},
            {"correct": True, "latency_ms": 30.0, "video_id": "v2", "ground_truth": "C", "prediction": "C"},
        ]
    )

    assert summary["ground_truth_distribution"] == {"A": 1, "B": 1, "C": 1, "D": 0}
    assert summary["prediction_distribution"] == {"A": 1, "B": 0, "C": 2, "D": 0}
    assert summary["accuracy_by_ground_truth_position"] == {
        "A": 1.0,
        "B": 0.0,
        "C": 1.0,
        "D": 0.0,
    }


def test_failure_store_persists_failed_samples(tmp_path: Path):
    from streaming_video_understanding.evaluation.failures import FailureStore

    path = tmp_path / "failures.jsonl"
    FailureStore(path).append_failure(
        {
            "sample_id": "s-failed",
            "config_id": "cfg",
            "error_type": "RuntimeError",
            "error": "synthetic failure",
        }
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["correct"] is False


def test_evaluator_rejects_future_frame_before_backend_call():
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


def test_evaluator_manifest_separates_recent_and_memory_frames():
    from PIL import Image

    from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
    from streaming_video_understanding.evaluation.engine import evaluate_sample
    from streaming_video_understanding.models.mock import MockVLMBackend
    from streaming_video_understanding.utils.config import (
        DatasetConfig,
        ExperimentConfig,
        ProjectConfig,
    )

    class StaticReader:
        def read_until(self, video_path: Path, end_time_s: float):
            return tuple(
                (Image.new("RGB", (8, 8)), FrameRecord(index, float(index)))
                for index in range(10)
            )

    sample = StreamingSample(
        sample_id="s2",
        video_id="v2",
        video_path=Path("video.mp4"),
        question="What is visible?",
        options=("cat", "dog", "ball", "car"),
        answer="A",
        query_time_s=9.0,
        task_type="Object Perception",
        temporal_clue_type="Prior",
    )
    record = evaluate_sample(
        sample,
        config=ProjectConfig(
            dataset=DatasetConfig(context_horizon_s=60.0),
            experiment=ExperimentConfig(visual_budget=4, seed=7),
        ),
        frame_reader=StaticReader(),
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        policy_id="recent-2+timestamp-aware-memory-2",
    )

    assert [frame["frame_id"] for frame in record["selected_recent_frames"]] == [8, 9]
    assert [frame["frame_id"] for frame in record["selected_memory_frames"]] == [0, 7]
