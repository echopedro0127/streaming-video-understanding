import json
import math
import time
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


def test_evaluate_sample_releases_query_images_after_inference():
    from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
    from streaming_video_understanding.evaluation.engine import evaluate_sample
    from streaming_video_understanding.models.mock import MockVLMBackend
    from streaming_video_understanding.utils.config import (
        DatasetConfig,
        ExperimentConfig,
        ProjectConfig,
    )

    class TrackingImage:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Reader:
        def __init__(self):
            self.images = [TrackingImage() for _ in range(3)]
            self.released = None

        def read_until(self, video_path, end_time_s):
            return iter(
                (image, FrameRecord(index, float(index)))
                for index, image in enumerate(self.images)
            )

        def release_query_images(self, video_path, images):
            self.released = tuple(images)
            for image in images:
                image.close()

    reader = Reader()
    evaluate_sample(
        StreamingSample(
            sample_id="cleanup-sample",
            video_id="cleanup-video",
            video_path=Path("video.mp4"),
            question="What is visible?",
            options=("cat", "dog", "ball", "car"),
            answer="A",
            query_time_s=2.0,
            task_type="Object Recognition",
            temporal_clue_type="Prior",
        ),
        config=ProjectConfig(
            dataset=DatasetConfig(context_horizon_s=60.0),
            experiment=ExperimentConfig(visual_budget=2, seed=7),
        ),
        frame_reader=reader,
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        policy_id="recent-2",
    )

    assert reader.released is not None
    assert len(reader.released) == 3
    assert all(image.closed for image in reader.images)


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
    assert [frame["frame_id"] for frame in record["selected_memory_frames"]] == [2, 5]
    assert record["retrieval_latency_ms"] == 0.0


def test_evaluator_semantic_policy_scores_history_only_and_keeps_budget_causal():
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

    class RecordingScorer:
        def __init__(self):
            self.calls = []

        def score_frames(self, **kwargs):
            self.calls.append(kwargs)
            return {frame.frame_id: float(frame.frame_id) for frame in kwargs["frames"]}

    scorer = RecordingScorer()
    backend = MockVLMBackend(answer="A")
    sample = StreamingSample(
        sample_id="semantic-sample",
        video_id="semantic-video",
        video_path=Path("video.mp4"),
        question="Which object is visible?",
        options=("cat", "dog", "ball", "car"),
        answer="A",
        query_time_s=9.0,
        task_type="Object Recognition",
        temporal_clue_type="Prior",
    )
    record = evaluate_sample(
        sample,
        config=ProjectConfig(
            dataset=DatasetConfig(context_horizon_s=60.0),
            experiment=ExperimentConfig(visual_budget=6, seed=7),
        ),
        frame_reader=StaticReader(),
        backend=backend,
        output_store=None,
        policy_id="recent-2+semantic-memory-4",
        semantic_scorer=scorer,
    )

    assert len(scorer.calls) == 1
    call = scorer.calls[0]
    assert set(call) == {"video_path", "frames", "images_by_id", "query"}
    assert [frame.frame_id for frame in call["frames"]] == list(range(8))
    assert sorted(call["images_by_id"]) == list(range(10))
    assert call["query"] == sample.question
    assert [frame["frame_id"] for frame in record["selected_memory_frames"]] == [4, 5, 6, 7]
    assert [frame["frame_id"] for frame in record["selected_recent_frames"]] == [8, 9]
    assert len(record["selected_frames"]) == 6
    assert record["timestamps_s"] == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert all(timestamp <= sample.query_time_s for timestamp in record["timestamps_s"])
    assert [item["frame_id"] for item in record["semantic_selection_trace"]] == [4, 5, 6, 7]
    assert [item["semantic_score"] for item in record["semantic_selection_trace"]] == [4.0, 5.0, 6.0, 7.0]
    assert backend.last_request is not None
    assert backend.last_request.include_timestamps is False
    assert record["retrieval_latency_ms"] >= 0.0


def test_evaluator_option_semantic_policy_scores_ordered_history_and_serializes_trace():
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

    class RecordingOptionScorer:
        def __init__(self):
            self.calls = []

        def score_option_frames(self, **kwargs):
            self.calls.append(kwargs)
            return {
                frame.frame_id: (
                    float(frame.frame_id),
                    float(frame.frame_id) + 0.1,
                    float(frame.frame_id) + 0.2,
                    float(frame.frame_id) + 0.3,
                )
                for frame in kwargs["frames"]
            }

    sample = StreamingSample(
        sample_id="option-semantic-sample",
        video_id="option-semantic-video",
        video_path=Path("video.mp4"),
        question="Which object is visible?",
        options=("cat", "dog", "ball", "car"),
        answer="ground truth answer",
        query_time_s=9.0,
        task_type="Object Recognition",
        temporal_clue_type="Prior",
    )
    scorer = RecordingOptionScorer()
    record = evaluate_sample(
        sample,
        config=ProjectConfig(
            dataset=DatasetConfig(context_horizon_s=60.0),
            experiment=ExperimentConfig(visual_budget=6, seed=7),
        ),
        frame_reader=StaticReader(),
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        policy_id="recent-2+option-semantic-memory-4",
        semantic_scorer=scorer,
    )

    assert len(scorer.calls) == 1
    call = scorer.calls[0]
    assert set(call) == {"video_path", "frames", "images_by_id", "question", "options"}
    assert [frame.frame_id for frame in call["frames"]] == list(range(8))
    assert sorted(call["images_by_id"]) == list(range(10))
    assert call["question"] == sample.question
    assert call["options"] == sample.options
    assert sample.answer not in call.values()

    assert [frame["frame_id"] for frame in record["selected_memory_frames"]] == [4, 5, 6, 7]
    assert [frame["frame_id"] for frame in record["selected_recent_frames"]] == [8, 9]
    assert len(record["selected_frames"]) == 6
    assert all(timestamp <= sample.query_time_s for timestamp in record["timestamps_s"])
    assert record["retrieval_latency_ms"] >= 0.0
    assert [item["frame_id"] for item in record["semantic_selection_trace"]] == [4, 5, 6, 7]
    assert all(
        set(item)
        == {
            "frame_id",
            "timestamp_s",
            "semantic_score",
            "objective_score",
            "option_scores",
            "best_option_index",
            "option_margin",
        }
        for item in record["semantic_selection_trace"]
    )
    for item in record["semantic_selection_trace"]:
        frame_id = item["frame_id"]
        assert item["option_scores"] == pytest.approx(
            [frame_id, frame_id + 0.1, frame_id + 0.2, frame_id + 0.3]
        )
        assert item["semantic_score"] == pytest.approx(frame_id + 0.3)
        assert item["objective_score"] == pytest.approx(frame_id + 0.3)
        assert item["best_option_index"] == 3
        assert item["option_margin"] == pytest.approx(0.1)


def test_evaluator_recent_6_option_memory_2_respects_budget_causality_and_trace_determinism():
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

    class RecordingOptionScorer:
        def __init__(self):
            self.calls = []

        def score_option_frames(self, **kwargs):
            self.calls.append(kwargs)
            return {
                frame.frame_id: (float(frame.frame_id), 0.0, 0.0, 0.0)
                for frame in kwargs["frames"]
            }

    sample = StreamingSample(
        sample_id="option-semantic-6plus2",
        video_id="video-6plus2",
        video_path=Path("video.mp4"),
        question="What happened before the person sat down?",
        options=("walked", "opened a door", "picked up a cup", "waved"),
        answer="opened a door",
        query_time_s=9.0,
        task_type="Temporal Reasoning",
        temporal_clue_type="Prior",
    )
    config = ProjectConfig(
        dataset=DatasetConfig(context_horizon_s=60.0),
        experiment=ExperimentConfig(visual_budget=8, seed=7),
    )

    first_scorer = RecordingOptionScorer()
    first = evaluate_sample(
        sample,
        config=config,
        frame_reader=StaticReader(),
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        policy_id="recent-6+option-semantic-memory-2",
        semantic_scorer=first_scorer,
    )
    second_scorer = RecordingOptionScorer()
    second = evaluate_sample(
        sample,
        config=config,
        frame_reader=StaticReader(),
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        policy_id="recent-6+option-semantic-memory-2",
        semantic_scorer=second_scorer,
    )

    assert len(first["selected_frames"]) <= 8
    assert len(first["selected_recent_frames"]) <= 6
    assert len(first["selected_memory_frames"]) <= 2
    assert all(
        frame["timestamp_s"] <= sample.query_time_s + 1e-6
        for frame in first["selected_frames"]
    )
    assert len(first_scorer.calls) == 1
    call = first_scorer.calls[0]
    assert call["question"] == sample.question
    assert call["options"] == sample.options
    assert "answer" not in call
    assert "ground_truth" not in call
    assert first["semantic_selection_trace"] == second["semantic_selection_trace"]


def test_evaluator_option_semantic_diversity_only_enables_diverse_selector(monkeypatch):
    from PIL import Image

    import streaming_video_understanding.evaluation.engine as engine
    from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
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
                for index in range(6)
            )

    class OptionScorer:
        def score_option_frames(self, **kwargs):
            return {
                frame.frame_id: (float(frame.frame_id), 0.0, 0.0, 0.0)
                for frame in kwargs["frames"]
            }

    selector_diversity = []
    original_selector = engine.select_semantic_history

    def recording_selector(*args, **kwargs):
        selector_diversity.append(kwargs["diversity"])
        return original_selector(*args, **kwargs)

    monkeypatch.setattr(engine, "select_semantic_history", recording_selector)
    sample = StreamingSample(
        sample_id="option-semantic-diversity",
        video_id="video",
        video_path=Path("video.mp4"),
        question="What is visible?",
        options=("cat", "dog", "ball", "car"),
        answer="A",
        query_time_s=5.0,
        task_type="Object Recognition",
        temporal_clue_type="Prior",
    )
    config = ProjectConfig(
        dataset=DatasetConfig(context_horizon_s=60.0),
        experiment=ExperimentConfig(visual_budget=4, seed=7),
    )

    for policy_id in (
        "recent-2+option-semantic-memory-2",
        "recent-2+option-semantic-diverse-memory-2",
    ):
        engine.evaluate_sample(
            sample,
            config=config,
            frame_reader=StaticReader(),
            backend=MockVLMBackend(answer="A"),
            output_store=None,
            policy_id=policy_id,
            semantic_scorer=OptionScorer(),
        )

    assert selector_diversity == [False, True]


@pytest.mark.parametrize("scorer", [None, object()])
def test_evaluator_option_semantic_policy_requires_option_scorer_before_backend_call(scorer):
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
            return ((Image.new("RGB", (8, 8)), FrameRecord(0, 0.0)),)

    backend = MockVLMBackend(answer="A")
    with pytest.raises(ValueError, match="score_option_frames"):
        evaluate_sample(
            StreamingSample(
                sample_id="missing-option-scorer",
                video_id="video",
                video_path=Path("video.mp4"),
                question="What is visible?",
                options=("cat", "dog", "ball", "car"),
                answer="A",
                query_time_s=0.0,
                task_type="Object Recognition",
                temporal_clue_type="Prior",
            ),
            config=ProjectConfig(
                dataset=DatasetConfig(context_horizon_s=60.0),
                experiment=ExperimentConfig(visual_budget=4, seed=7),
            ),
            frame_reader=StaticReader(),
            backend=backend,
            output_store=None,
            policy_id="recent-2+option-semantic-memory-2",
            semantic_scorer=scorer,
        )

    assert backend.call_count == 0


def test_evaluator_rejects_option_policy_over_visual_budget_before_decode_or_scoring():
    from PIL import Image

    from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
    from streaming_video_understanding.evaluation.engine import evaluate_sample
    from streaming_video_understanding.models.mock import MockVLMBackend
    from streaming_video_understanding.utils.config import (
        DatasetConfig,
        ExperimentConfig,
        ProjectConfig,
    )

    class RecordingReader:
        def __init__(self):
            self.call_count = 0

        def read_until(self, video_path: Path, end_time_s: float):
            self.call_count += 1
            return ((Image.new("RGB", (8, 8)), FrameRecord(0, 0.0)),)

    class RecordingOptionScorer:
        def __init__(self):
            self.call_count = 0

        def score_option_frames(self, **kwargs):
            self.call_count += 1
            return {}

    reader = RecordingReader()
    scorer = RecordingOptionScorer()
    backend = MockVLMBackend(answer="A")
    with pytest.raises(ValueError, match="visual_budget|budget"):
        evaluate_sample(
            StreamingSample(
                sample_id="option-over-budget",
                video_id="video",
                video_path=Path("video.mp4"),
                question="What is visible?",
                options=("cat", "dog", "ball", "car"),
                answer="A",
                query_time_s=9.0,
                task_type="Object Recognition",
                temporal_clue_type="Prior",
            ),
            config=ProjectConfig(
                dataset=DatasetConfig(context_horizon_s=60.0),
                experiment=ExperimentConfig(visual_budget=4, seed=7),
            ),
            frame_reader=reader,
            backend=backend,
            output_store=None,
            policy_id="recent-3+option-semantic-memory-3",
            semantic_scorer=scorer,
        )

    assert reader.call_count == 0
    assert scorer.call_count == 0
    assert backend.call_count == 0


@pytest.mark.parametrize(
    ("policy_id", "expected"),
    [
        ("recent", ("recent", 8, 0)),
        ("uniform", ("uniform", 0, 8)),
        ("recent-2", ("recent", 2, 0)),
        ("uniform-2", ("uniform", 0, 2)),
        ("recent-2+uniform-memory-3", ("uniform", 2, 3)),
        ("recent-2+timestamp-aware-memory-3", ("timestamp-aware", 2, 3)),
        ("recent-2+semantic-memory-3", ("semantic", 2, 3)),
        ("recent-2+semantic-diverse-memory-3", ("semantic-diverse", 2, 3)),
    ],
)
def test_policy_parts_preserves_existing_policy_ids_and_kinds(policy_id, expected):
    from streaming_video_understanding.evaluation.engine import _policy_parts

    assert _policy_parts(policy_id, budget=8) == expected


def test_policy_parts_accepts_recent_6_plus_option_memory_2():
    from streaming_video_understanding.evaluation.engine import _policy_parts

    assert _policy_parts("recent-6+option-semantic-memory-2", budget=8) == (
        "option-semantic",
        6,
        2,
    )


def test_evaluator_semantic_selection_does_not_depend_on_answer():
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

    class RecordingScorer:
        def score_frames(self, *, video_path, frames, images_by_id, query):
            return {frame.frame_id: float(frame.frame_id) for frame in frames}

    common = dict(
        sample_id="answer-independent",
        video_id="answer-video",
        video_path=Path("video.mp4"),
        question="Which object is visible?",
        options=("cat", "dog", "ball", "car"),
        query_time_s=9.0,
        task_type="Object Recognition",
        temporal_clue_type="Prior",
    )
    config = ProjectConfig(
        dataset=DatasetConfig(context_horizon_s=60.0),
        experiment=ExperimentConfig(visual_budget=6, seed=7),
    )
    first = evaluate_sample(
        StreamingSample(answer="A", **common),
        config=config,
        frame_reader=StaticReader(),
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        policy_id="recent-2+semantic-memory-4",
        semantic_scorer=RecordingScorer(),
    )
    second = evaluate_sample(
        StreamingSample(answer="B", **common),
        config=config,
        frame_reader=StaticReader(),
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        policy_id="recent-2+semantic-memory-4",
        semantic_scorer=RecordingScorer(),
    )

    assert first["selected_memory_frames"] == second["selected_memory_frames"]
    assert first["semantic_selection_trace"] == second["semantic_selection_trace"]


def test_evaluator_semantic_policy_requires_scorer_before_backend_call():
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
            return ((Image.new("RGB", (8, 8)), FrameRecord(0, 0.0)),)

    backend = MockVLMBackend(answer="A")
    with pytest.raises(ValueError, match="requires semantic_scorer"):
        evaluate_sample(
            StreamingSample(
                sample_id="missing-scorer",
                video_id="video",
                video_path=Path("video.mp4"),
                question="What is visible?",
                options=("cat", "dog", "ball", "car"),
                answer="A",
                query_time_s=0.0,
                task_type="Object Recognition",
                temporal_clue_type="Prior",
            ),
            config=ProjectConfig(
                dataset=DatasetConfig(context_horizon_s=60.0),
                experiment=ExperimentConfig(visual_budget=4, seed=7),
            ),
            frame_reader=StaticReader(),
            backend=backend,
            output_store=None,
            policy_id="recent-2+semantic-memory-2",
        )

    assert backend.call_count == 0


def test_evaluator_rejects_combined_policy_over_visual_budget_before_work():
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

    class RecordingScorer:
        def __init__(self):
            self.call_count = 0

        def score_frames(self, *, video_path, frames, images_by_id, query):
            self.call_count += 1
            return {frame.frame_id: float(frame.frame_id) for frame in frames}

    backend = MockVLMBackend(answer="A")
    scorer = RecordingScorer()
    with pytest.raises(ValueError, match="visual_budget|budget"):
        evaluate_sample(
            StreamingSample(
                sample_id="over-budget",
                video_id="video",
                video_path=Path("video.mp4"),
                question="What is visible?",
                options=("cat", "dog", "ball", "car"),
                answer="A",
                query_time_s=9.0,
                task_type="Object Recognition",
                temporal_clue_type="Prior",
            ),
            config=ProjectConfig(
                dataset=DatasetConfig(context_horizon_s=60.0),
                experiment=ExperimentConfig(visual_budget=4, seed=7),
            ),
            frame_reader=StaticReader(),
            backend=backend,
            output_store=None,
            policy_id="recent-3+semantic-memory-3",
            semantic_scorer=scorer,
        )

    assert backend.call_count == 0
    assert scorer.call_count == 0


@pytest.mark.parametrize(
    "policy_id",
    ["recent-2+semantic-memory-2", "recent-2+semantic-diverse-memory-2"],
)
def test_evaluator_accepts_semantic_policy_ids(policy_id: str):
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
                for index in range(6)
            )

    class RecordingScorer:
        def score_frames(self, *, video_path, frames, images_by_id, query):
            return {frame.frame_id: float(frame.frame_id) for frame in frames}

    record = evaluate_sample(
        StreamingSample(
            sample_id="semantic-policy",
            video_id="video",
            video_path=Path("video.mp4"),
            question="What is visible?",
            options=("cat", "dog", "ball", "car"),
            answer="A",
            query_time_s=5.0,
            task_type="Object Recognition",
            temporal_clue_type="Prior",
        ),
        config=ProjectConfig(
            dataset=DatasetConfig(context_horizon_s=60.0),
            experiment=ExperimentConfig(visual_budget=4, seed=7),
        ),
        frame_reader=StaticReader(),
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        policy_id=policy_id,
        semantic_scorer=RecordingScorer(),
    )

    assert len(record["selected_frames"]) == 4


def test_evaluator_writes_stage_level_profile(tmp_path: Path):
    from PIL import Image

    from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
    from streaming_video_understanding.evaluation.engine import evaluate_sample
    from streaming_video_understanding.evaluation.profiling import RuntimeProfileStore
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
                for index in range(6)
            )

    profile_path = tmp_path / "profile.jsonl"
    record = evaluate_sample(
        StreamingSample(
            sample_id="profile-sample",
            video_id="profile-video",
            video_path=Path("video.mp4"),
            question="What is visible?",
            options=("cat", "dog", "ball", "car"),
            answer="A",
            query_time_s=5.0,
            task_type="Object Recognition",
            temporal_clue_type="Prior",
            frames_required="single",
        ),
        config=ProjectConfig(
            dataset=DatasetConfig(context_horizon_s=60.0),
            experiment=ExperimentConfig(visual_budget=4, seed=7),
        ),
        frame_reader=StaticReader(),
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        profile_store=RuntimeProfileStore(profile_path),
    )

    assert "latency_ms" in record
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert {
        "decode_ms",
        "retrieval_ms",
        "sampling_ms",
        "memory_ms",
        "processor_ms",
        "generation_ms",
        "write_ms",
        "total_ms",
    } <= profile.keys()
    assert profile["retrieval_ms"] == 0.0
    assert profile["decode_ms"] >= 0.0
    assert profile["total_ms"] >= profile["decode_ms"]


def test_evaluator_profiles_semantic_retrieval_separately_from_sampling(tmp_path: Path):
    from PIL import Image

    from streaming_video_understanding.data.schema import FrameRecord, StreamingSample
    from streaming_video_understanding.evaluation.engine import evaluate_sample
    from streaming_video_understanding.evaluation.profiling import RuntimeProfileStore
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
                for index in range(6)
            )

    class SlowScorer:
        def score_frames(self, *, video_path, frames, images_by_id, query):
            time.sleep(0.05)
            return {frame.frame_id: float(frame.frame_id) for frame in frames}

    profile_path = tmp_path / "semantic-profile.jsonl"
    record = evaluate_sample(
        StreamingSample(
            sample_id="semantic-profile-sample",
            video_id="semantic-profile-video",
            video_path=Path("video.mp4"),
            question="What is visible?",
            options=("cat", "dog", "ball", "car"),
            answer="A",
            query_time_s=5.0,
            task_type="Object Recognition",
            temporal_clue_type="Prior",
        ),
        config=ProjectConfig(
            dataset=DatasetConfig(context_horizon_s=60.0),
            experiment=ExperimentConfig(visual_budget=4, seed=7),
        ),
        frame_reader=StaticReader(),
        backend=MockVLMBackend(answer="A"),
        output_store=None,
        policy_id="recent-2+semantic-memory-2",
        semantic_scorer=SlowScorer(),
        profile_store=RuntimeProfileStore(profile_path),
    )

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert math.isfinite(profile["retrieval_ms"])
    assert math.isfinite(profile["sampling_ms"])
    assert profile["retrieval_ms"] >= 30.0
    assert profile["sampling_ms"] < profile["retrieval_ms"]
    assert record["retrieval_latency_ms"] == profile["retrieval_ms"]


def test_metrics_summarize_retrieval_stage_timing():
    summary = summarize_records(
        [
            {"retrieval_ms": 0.0},
            {"retrieval_ms": 4.0},
        ]
    )

    assert summary["stage_timing"]["retrieval_ms"] == {
        "mean_ms": 2.0,
        "p50_ms": 2.0,
        "p95_ms": 3.8,
    }
