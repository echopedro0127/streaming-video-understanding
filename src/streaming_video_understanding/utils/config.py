from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    context_horizon_s: float
    timestamp_tolerance_s: float = 1e-6


@dataclass(frozen=True)
class ExperimentConfig:
    visual_budget: int
    seed: int


@dataclass(frozen=True)
class ProjectConfig:
    dataset: DatasetConfig
    experiment: ExperimentConfig


def _mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a mapping")
    return data


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset = _mapping(raw.get("dataset"), "dataset")
    experiment = _mapping(raw.get("experiment"), "experiment")

    try:
        context_horizon_s = float(dataset["context_horizon_s"])
        timestamp_tolerance_s = float(dataset.get("timestamp_tolerance_s", 1e-6))
        visual_budget = int(experiment["visual_budget"])
        seed = int(experiment["seed"])
    except KeyError as exc:
        raise ValueError(f"Missing required configuration key: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid configuration values in {config_path}") from exc

    if context_horizon_s <= 0:
        raise ValueError("context_horizon_s must be positive")
    if timestamp_tolerance_s < 0:
        raise ValueError("timestamp_tolerance_s must be non-negative")
    if visual_budget <= 0:
        raise ValueError("visual_budget must be positive")

    return ProjectConfig(
        dataset=DatasetConfig(
            context_horizon_s=context_horizon_s,
            timestamp_tolerance_s=timestamp_tolerance_s,
        ),
        experiment=ExperimentConfig(visual_budget=visual_budget, seed=seed),
    )
