import sys
import types
from pathlib import Path

import pytest

from scripts.run_eval import _requires_semantic_scorer
from streaming_video_understanding.utils.config import load_config


def test_checked_in_configs_are_loadable():
    config_paths = sorted(Path("configs").glob("*.yaml"))
    assert config_paths
    for path in config_paths:
        config = load_config(path)
        assert config.dataset.context_horizon_s == 60.0
        assert config.experiment.visual_budget in {4, 8, 16}


def test_semantic_validation_config_declares_expected_experiment():
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


def test_option_semantic_validation_config_declares_expected_experiment():
    baseline = load_config(Path("configs/semantic-validation-1fps.yaml"))
    config = load_config(Path("configs/option-semantic-validation-1fps.yaml"))

    assert config.dataset == baseline.dataset
    assert config.model == baseline.model
    assert config.runtime == baseline.runtime
    assert config.retrieval == baseline.retrieval
    assert config.experiment.visual_budget == 8
    assert config.experiment.policy_ids == (
        "recent-4+option-semantic-memory-4",
        "recent-4+option-semantic-diverse-memory-4",
    )


@pytest.mark.parametrize(
    ("policy_id", "expected"),
    [
        ("recent-8", False),
        ("uniform-8", False),
        ("recent-4+semantic-memory-4", True),
        ("RECENT-4+OPTION-SEMANTIC-MEMORY-4", True),
        ("recent-4+semanticish-memory-4", False),
    ],
)
def test_requires_semantic_scorer_for_exact_policy_kinds(policy_id, expected):
    assert _requires_semantic_scorer((policy_id,)) is expected


def test_main_constructs_one_scorer_and_reuses_it(monkeypatch, tmp_path):
    import scripts.run_eval as run_eval

    constructed = []

    class FakeClipFrameScorer:
        def __init__(self, **kwargs):
            constructed.append(self)

    fake_clip_module = types.ModuleType("streaming_video_understanding.retrieval.clip")
    fake_clip_module.ClipFrameScorer = FakeClipFrameScorer
    monkeypatch.setitem(
        sys.modules,
        "streaming_video_understanding.retrieval.clip",
        fake_clip_module,
    )

    passed_scorers = []
    monkeypatch.setattr(
        run_eval,
        "evaluate_sample",
        lambda *args, **kwargs: passed_scorers.append(kwargs["semantic_scorer"]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--config",
            "configs/option-semantic-validation-1fps.yaml",
            "--backend",
            "mock",
            "--limit",
            "2",
            "--policy",
            "recent-4+option-semantic-memory-4",
            "--output",
            str(tmp_path / "output.jsonl"),
            "--overwrite",
        ],
    )

    assert run_eval.main() == 0
    assert len(constructed) == 1
    assert len(passed_scorers) == 2
    assert all(scorer is constructed[0] for scorer in passed_scorers)
