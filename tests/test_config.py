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
