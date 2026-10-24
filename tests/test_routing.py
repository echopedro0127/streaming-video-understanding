from dataclasses import replace
from pathlib import Path

import pytest

from streaming_video_understanding.streaming.routing import route_question


@pytest.mark.parametrize(
    "question,group,resolved,allocation",
    [
        ("What is happening right now?", "current", "recent-8", (8, 0)),
        ("Which object is currently on the table?", "current", "recent-8", (8, 0)),
        ("What happened before he sat down?", "history", "recent-4+semantic-memory-4", (4, 4)),
        ("What was initially on the table?", "history", "recent-4+semantic-memory-4", (4, 4)),
        ("What did she do earlier, compared with now?", "history", "recent-4+semantic-memory-4", (4, 4)),
        ("How many times has she waved so far?", "counting", "recent-4+semantic-diverse-memory-4", (4, 4)),
        ("How many cups are visible now?", "current", "recent-8", (8, 0)),
        ("What is on the table?", "unspecified", "recent-6+semantic-memory-2", (6, 2)),
        ("Identify the aftershave bottle.", "unspecified", "recent-6+semantic-memory-2", (6, 2)),
        ("Who is in the show nowadays?", "unspecified", "recent-6+semantic-memory-2", (6, 2)),
    ],
)
def test_question_only_routing(question, group, resolved, allocation):
    route = route_question(question, budget=8)
    assert route.group == group
    assert route.resolved_policy_id == resolved
    assert (route.recent_capacity, route.memory_capacity) == allocation
    assert route.version == "question-rules-v1"
    assert route.reason


@pytest.mark.parametrize("question,budget", [("", 8), ("  ", 8), ("What?", 7), ("What?", True)])
def test_rejects_undefined_router_inputs(question, budget):
    with pytest.raises(ValueError):
        route_question(question, budget=budget)


def _sample(question):
    from streaming_video_understanding.data.schema import StreamingSample
    return StreamingSample(
        sample_id="route-test", video_id="sample_1", video_path=Path("video.mp4"),
        question=question, options=("cat", "dog", "ball", "car"), answer="A",
        query_time_s=19.0, task_type="unused", temporal_clue_type="unused",
    )


def _config():
    from streaming_video_understanding.utils.config import DatasetConfig, ExperimentConfig, ProjectConfig
    return ProjectConfig(dataset=DatasetConfig(context_horizon_s=60.0),
                         experiment=ExperimentConfig(visual_budget=8, seed=42))


class Reader:
    def read_until(self, video_path, end_time_s):
        from PIL import Image
        from streaming_video_understanding.data.schema import FrameRecord
        for index in range(20):
            yield Image.new("RGB", (4, 4)), FrameRecord(index, float(index))


class Scorer:
    def score_frames(self, *, video_path, frames, images_by_id, query):
        return {frame.frame_id: -float(frame.frame_id) for frame in frames}


@pytest.mark.parametrize("question,k,m", [("What is visible now?", 8, 0),
                                         ("What happened before?", 4, 4),
                                         ("What is visible?", 6, 2),
                                         ("How many times did he jump?", 4, 4)])
def test_evaluator_records_route_without_changing_policy_identity(question, k, m):
    from streaming_video_understanding.evaluation.engine import evaluate_sample
    from streaming_video_understanding.models.mock import MockVLMBackend
    # Running both answers/option orders verifies labels cannot influence selection.
    sample = _sample(question)
    records = []
    for item in (sample, replace(sample, answer="D", options=tuple(reversed(sample.options)), task_type="changed")):
        records.append(evaluate_sample(
            item, config=_config(), frame_reader=Reader(), backend=MockVLMBackend("A"),
            output_store=None, policy_id="adaptive-semantic-v1",
            semantic_scorer=Scorer() if m else None,
        ))
    first, second = records
    assert first["policy_id"] == "adaptive-semantic-v1"
    assert first["routing"]["recent_capacity"] == k
    assert first["routing"]["memory_capacity"] == m
    assert first["selected_frames"] == second["selected_frames"]
    assert len(first["selected_recent_frames"]) == k
    assert len(first["selected_memory_frames"]) == m
    assert first["input_frames"] == 8
    assert max(first["timestamps_s"]) <= sample.query_time_s
    assert first["task_type"] == "unused"
    assert first["context_start_s"] == 0.0
    assert first["context_horizon_s"] == 60.0


def test_adaptive_history_requires_scorer():
    from streaming_video_understanding.evaluation.engine import evaluate_sample
    from streaming_video_understanding.models.mock import MockVLMBackend
    with pytest.raises(ValueError, match="requires semantic_scorer"):
        evaluate_sample(_sample("What happened earlier?"), config=_config(),
                        frame_reader=Reader(), backend=MockVLMBackend("A"),
                        output_store=None, policy_id="adaptive-semantic-v1")


def test_cli_initializes_scorer_for_adaptive_policy():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_eval_router", Path(__file__).parents[1] / "scripts/run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._requires_semantic_scorer(("adaptive-semantic-v1",))
    assert not module._requires_semantic_scorer(("recent-8",))


def test_adaptive_route_still_rejects_future_frames():
    from streaming_video_understanding.evaluation.engine import evaluate_sample
    from streaming_video_understanding.models.mock import MockVLMBackend
    from streaming_video_understanding.streaming.buffer import FutureFrameLeakageError
    with pytest.raises(FutureFrameLeakageError):
        evaluate_sample(replace(_sample("What is visible now?"), query_time_s=18.0),
                        config=_config(), frame_reader=Reader(), backend=MockVLMBackend("A"),
                        output_store=None, policy_id="adaptive-semantic-v1")


def test_adaptive_config_has_matched_q_only_controls():
    from streaming_video_understanding.utils.config import load_config
    config = load_config(Path(__file__).parents[1] / "configs/adaptive-semantic-1fps.yaml")
    assert config.dataset.context_horizon_s == 60.0
    assert config.experiment.visual_budget == 8
    assert set(config.experiment.policy_ids) == {
        "recent-8", "recent-6+semantic-memory-2", "recent-4+semantic-memory-4",
        "recent-4+semantic-diverse-memory-4", "adaptive-semantic-v1",
    }


def test_cli_mock_adaptive_resume_does_not_duplicate_completed_keys(monkeypatch, tmp_path):
    import json
    import sys
    import types
    import scripts.run_eval as cli

    fake_clip = types.ModuleType("streaming_video_understanding.retrieval.clip")
    class FakeClip(Scorer):
        def __init__(self, **kwargs):
            pass
    fake_clip.ClipFrameScorer = FakeClip
    monkeypatch.setitem(sys.modules, "streaming_video_understanding.retrieval.clip", fake_clip)
    path = tmp_path / "adaptive.jsonl"
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--config", "configs/adaptive-semantic-1fps.yaml",
                                     "--backend", "mock", "--policy", "adaptive-semantic-v1",
                                     "--output", str(path), "--resume"])
    assert cli.main() == 0
    before = path.read_bytes()
    assert cli.main() == 0
    assert path.read_bytes() == before
    rows = [json.loads(line) for line in before.decode().splitlines()]
    assert len(rows) == 3
    assert all(row["policy_id"] == "adaptive-semantic-v1" and row["input_frames"] <= 8 for row in rows)
    assert all(row["routing"]["version"] == "question-rules-v1" for row in rows)
