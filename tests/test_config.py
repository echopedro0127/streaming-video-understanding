from pathlib import Path

import pytest

from streaming_video_understanding.utils.config import configuration_id, load_config


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


def test_load_config_preserves_policy_ids(tmp_path: Path):
    config_path = tmp_path / "policies.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 8\n  seed: 7\n"
        "  policy_ids: [recent-8, uniform-8]\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.experiment.policy_ids == ("recent-8", "uniform-8")


def test_load_config_preserves_runtime_and_model_settings(tmp_path: Path):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 8\n  seed: 7\n"
        "model:\n  backend: mock\n  model_id: test-model\n  max_new_tokens: 4\n"
        "runtime:\n  sample_fps: 2\n  batch_size: 1\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model.backend == "mock"
    assert config.model.model_id == "test-model"
    assert config.model.max_new_tokens == 4
    assert config.runtime.sample_fps == 2.0
    assert configuration_id(config, policy_id="recent-8", backend="mock")


def test_load_config_rejects_non_finite_runtime(tmp_path: Path):
    config_path = tmp_path / "invalid-runtime.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 8\n  seed: 7\n"
        "runtime:\n  sample_fps: .nan\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_fps"):
        load_config(config_path)
