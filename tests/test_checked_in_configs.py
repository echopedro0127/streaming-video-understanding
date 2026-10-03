from pathlib import Path

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


def test_requires_semantic_scorer_for_semantic_policies_only():
    assert _requires_semantic_scorer(("recent-8", "uniform-8")) is False
    assert _requires_semantic_scorer(("recent-4+semantic-memory-4",)) is True
