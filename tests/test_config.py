from dataclasses import replace
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


def test_load_config_uses_retrieval_defaults(tmp_path: Path):
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


def test_load_config_parses_retrieval_values_to_declared_types(tmp_path: Path):
    config_path = tmp_path / "retrieval.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 8\n  seed: 7\n"
        "retrieval:\n"
        "  model_id: custom/clip\n"
        "  device: cuda\n"
        "  cache_dir: custom-cache\n"
        "  image_batch_size: '32'\n"
        "  candidate_multiplier: '6'\n"
        "  cluster_window_s: '2.5'\n"
        "  diversity_weight: '0.5'\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.retrieval.model_id == "custom/clip"
    assert config.retrieval.device == "cuda"
    assert config.retrieval.cache_dir == "custom-cache"
    assert config.retrieval.image_batch_size == 32
    assert config.retrieval.candidate_multiplier == 6
    assert config.retrieval.cluster_window_s == 2.5
    assert config.retrieval.diversity_weight == 0.5


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
    config_path = tmp_path / f"invalid-{field}.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 8\n  seed: 7\n"
        f"retrieval:\n  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_config(config_path)


@pytest.mark.parametrize(
    ("field", "yaml_value"),
    [
        ("model_id", "''"),
        ("device", "metal"),
        ("cache_dir", "''"),
        ("cluster_window_s", ".nan"),
        ("diversity_weight", ".nan"),
    ],
)
def test_load_config_rejects_other_invalid_retrieval_values(
    tmp_path: Path, field: str, yaml_value: str
):
    config_path = tmp_path / f"invalid-{field}.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 8\n  seed: 7\n"
        f"retrieval:\n  {field}: {yaml_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_config(config_path)


def test_configuration_id_includes_retrieval_config(tmp_path: Path):
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(
        "dataset:\n  context_horizon_s: 60\n"
        "experiment:\n  visual_budget: 8\n  seed: 7\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    changed_config = replace(
        config,
        retrieval=replace(config.retrieval, image_batch_size=32),
    )

    assert configuration_id(
        config, policy_id="recent-8", backend="mock"
    ) != configuration_id(changed_config, policy_id="recent-8", backend="mock")
