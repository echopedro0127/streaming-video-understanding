from pathlib import Path

from streaming_video_understanding.utils.config import load_config


def test_checked_in_configs_are_loadable():
    config_paths = sorted(Path("configs").glob("*.yaml"))
    assert config_paths
    for path in config_paths:
        config = load_config(path)
        assert config.dataset.context_horizon_s == 60.0
        assert config.experiment.visual_budget in {4, 8, 16}
