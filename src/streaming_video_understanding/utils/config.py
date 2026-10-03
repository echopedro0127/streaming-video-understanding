from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    context_horizon_s: float
    timestamp_tolerance_s: float = 1e-6
    name: str = "StreamingBench"
    subset: str = "real-time-1-50"


@dataclass(frozen=True)
class ExperimentConfig:
    visual_budget: int
    seed: int
    policy_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "smolvlm"
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    max_new_tokens: int = 8


@dataclass(frozen=True)
class RuntimeConfig:
    sample_fps: float = 1.0
    batch_size: int = 1


@dataclass(frozen=True)
class RetrievalConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    cache_dir: str = ".cache/semantic_features"
    image_batch_size: int = 16
    candidate_multiplier: int = 4
    cluster_window_s: float = 3.0
    diversity_weight: float = 0.25


@dataclass(frozen=True)
class ProjectConfig:
    dataset: DatasetConfig
    experiment: ExperimentConfig
    model: ModelConfig = ModelConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    retrieval: RetrievalConfig = RetrievalConfig()


def _mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a mapping")
    return data


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    raise ValueError(f"{name} must be an integer")


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset = _mapping(raw.get("dataset"), "dataset")
    experiment = _mapping(raw.get("experiment"), "experiment")
    model = _mapping(raw.get("model", {}), "model")
    runtime = _mapping(raw.get("runtime", {}), "runtime")
    retrieval = _mapping(raw.get("retrieval", {}), "retrieval")

    raw_retrieval_model_id = retrieval.get(
        "model_id", "openai/clip-vit-base-patch32"
    )
    if raw_retrieval_model_id is None:
        raise ValueError("retrieval.model_id must be non-empty")
    raw_retrieval_cache_dir = retrieval.get(
        "cache_dir", ".cache/semantic_features"
    )
    if raw_retrieval_cache_dir is None:
        raise ValueError("retrieval.cache_dir must be non-empty")
    image_batch_size = _strict_int(
        retrieval.get("image_batch_size", 16), "retrieval.image_batch_size"
    )
    candidate_multiplier = _strict_int(
        retrieval.get("candidate_multiplier", 4), "retrieval.candidate_multiplier"
    )

    try:
        context_horizon_s = float(dataset["context_horizon_s"])
        timestamp_tolerance_s = float(dataset.get("timestamp_tolerance_s", 1e-6))
        visual_budget = int(experiment["visual_budget"])
        seed = int(experiment["seed"])
        raw_policy_ids = experiment.get("policy_ids", ())
        dataset_name = str(dataset.get("name", "StreamingBench")).strip()
        dataset_subset = str(dataset.get("subset", "real-time-1-50")).strip()
        model_backend = str(model.get("backend", "smolvlm")).strip().lower()
        model_id = str(
            model.get("model_id", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        ).strip()
        max_new_tokens = int(model.get("max_new_tokens", 8))
        sample_fps = float(runtime.get("sample_fps", 1.0))
        batch_size = int(runtime.get("batch_size", 1))
        retrieval_model_id = str(raw_retrieval_model_id).strip()
        retrieval_device = str(retrieval.get("device", "auto")).strip().lower()
        retrieval_cache_dir = str(raw_retrieval_cache_dir).strip()
        cluster_window_s = float(retrieval.get("cluster_window_s", 3.0))
        diversity_weight = float(retrieval.get("diversity_weight", 0.25))
    except KeyError as exc:
        raise ValueError(f"Missing required configuration key: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid configuration values in {config_path}") from exc

    if context_horizon_s <= 0:
        raise ValueError("context_horizon_s must be positive")
    if not math.isfinite(context_horizon_s):
        raise ValueError("context_horizon_s must be finite")
    if timestamp_tolerance_s < 0:
        raise ValueError("timestamp_tolerance_s must be non-negative")
    if not math.isfinite(timestamp_tolerance_s):
        raise ValueError("timestamp_tolerance_s must be finite")
    if visual_budget <= 0:
        raise ValueError("visual_budget must be positive")
    if model_backend not in {"mock", "smolvlm"}:
        raise ValueError("model.backend must be mock or smolvlm")
    if not model_id:
        raise ValueError("model.model_id must be non-empty")
    if max_new_tokens <= 0:
        raise ValueError("model.max_new_tokens must be positive")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("runtime.sample_fps must be a finite positive number")
    if batch_size <= 0:
        raise ValueError("runtime.batch_size must be positive")
    if not retrieval_model_id:
        raise ValueError("retrieval.model_id must be non-empty")
    if retrieval_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("retrieval.device must be auto, cpu, or cuda")
    if not retrieval_cache_dir:
        raise ValueError("retrieval.cache_dir must be non-empty")
    if image_batch_size <= 0:
        raise ValueError("retrieval.image_batch_size must be positive")
    if candidate_multiplier <= 0:
        raise ValueError("retrieval.candidate_multiplier must be positive")
    if not math.isfinite(cluster_window_s) or cluster_window_s <= 0:
        raise ValueError("retrieval.cluster_window_s must be a finite positive number")
    if not math.isfinite(diversity_weight) or not 0 <= diversity_weight <= 1:
        raise ValueError("retrieval.diversity_weight must be finite and within [0, 1]")
    if not isinstance(raw_policy_ids, (list, tuple)):
        raise ValueError("policy_ids must be a sequence")
    policy_ids = tuple(str(policy_id).strip() for policy_id in raw_policy_ids)
    if any(not policy_id for policy_id in policy_ids):
        raise ValueError("policy_ids must contain non-empty strings")

    return ProjectConfig(
        dataset=DatasetConfig(
            context_horizon_s=context_horizon_s,
            timestamp_tolerance_s=timestamp_tolerance_s,
            name=dataset_name,
            subset=dataset_subset,
        ),
        experiment=ExperimentConfig(
            visual_budget=visual_budget,
            seed=seed,
            policy_ids=policy_ids,
        ),
        model=ModelConfig(
            backend=model_backend,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        ),
        runtime=RuntimeConfig(sample_fps=sample_fps, batch_size=batch_size),
        retrieval=RetrievalConfig(
            model_id=retrieval_model_id,
            device=retrieval_device,
            cache_dir=retrieval_cache_dir,
            image_batch_size=image_batch_size,
            candidate_multiplier=candidate_multiplier,
            cluster_window_s=cluster_window_s,
            diversity_weight=diversity_weight,
        ),
    )


def configuration_id(config: ProjectConfig, *, policy_id: str, backend: str) -> str:
    payload = {
        "config": {
            "dataset": config.dataset.__dict__,
            "experiment": config.experiment.__dict__,
            "model": config.model.__dict__,
            "runtime": config.runtime.__dict__,
            "retrieval": config.retrieval.__dict__,
        },
        "policy_id": policy_id,
        "backend": backend,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"{policy_id}-{digest}"
