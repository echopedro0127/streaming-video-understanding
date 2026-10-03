from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..data.schema import FrameRecord
from .cache import FrameFeatureCache

if TYPE_CHECKING:
    from PIL import Image


def _feature_matrix(
    output: object,
    *,
    torch: object,
    output_name: str,
    expected_batch_size: int,
) -> np.ndarray:
    if torch.is_tensor(output):
        feature = output
    elif hasattr(output, "pooler_output"):
        feature = output.pooler_output
        if not torch.is_tensor(feature):
            raise RuntimeError(
                f"CLIP {output_name} output has an invalid pooler_output"
            )
    elif isinstance(output, (tuple, list)):
        candidates = [
            value
            for value in output
            if torch.is_tensor(value) and getattr(value, "ndim", None) == 2
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"CLIP {output_name} output does not contain one "
                "unambiguous feature tensor"
            )
        feature = candidates[0]
    else:
        raise RuntimeError(
            f"CLIP {output_name} output does not contain a feature tensor"
        )

    if getattr(feature, "ndim", None) != 2:
        raise RuntimeError(
            f"CLIP {output_name} output feature must have shape [batch, dim]"
        )
    converted = np.asarray(
        feature.float().cpu().numpy(), dtype=np.float32
    )
    if (
        converted.ndim != 2
        or converted.shape[0] != expected_batch_size
        or converted.shape[1] <= 0
    ):
        raise RuntimeError(
            f"CLIP {output_name} output feature must have shape "
            f"[{expected_batch_size}, dim] with a positive dimension"
        )
    return converted


class ClipFrameScorer:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        cache_dir: str | Path,
        image_batch_size: int,
    ) -> None:
        if not isinstance(model_id, str):
            raise TypeError("model_id must be a non-empty string")
        normalized_model_id = model_id.strip()
        if not normalized_model_id:
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(device, str):
            raise TypeError("device must be one of: auto, cpu, cuda")
        normalized_device = device.strip().lower()
        if normalized_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if type(image_batch_size) is not int or image_batch_size <= 0:
            raise ValueError("image_batch_size must be a positive exact integer")

        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError(
                "ClipFrameScorer requires torch and transformers"
            ) from exc

        cuda_available = torch.cuda.is_available()
        if normalized_device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested but is not available")
        resolved_device = (
            "cuda"
            if normalized_device == "cuda"
            or (normalized_device == "auto" and cuda_available)
            else "cpu"
        )
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32

        self._torch = torch
        self._device = resolved_device
        self._image_batch_size = image_batch_size
        self._processor = AutoProcessor.from_pretrained(normalized_model_id)
        model = CLIPModel.from_pretrained(
            normalized_model_id, torch_dtype=dtype
        )
        self._model = model.to(resolved_device)
        self._model.eval()
        self._cache = FrameFeatureCache(
            cache_dir, model_id=normalized_model_id
        )

    def _encode_text(self, query: str) -> np.ndarray:
        inputs = self._processor(
            text=[query], return_tensors="pt", padding=True
        )
        device_inputs = {
            name: inputs[name].to(self._device)
            if hasattr(inputs[name], "to")
            else inputs[name]
            for name in ("input_ids", "attention_mask", "position_ids")
            if name in inputs
        }
        with self._torch.inference_mode():
            output = self._model.get_text_features(**device_inputs)
        return _feature_matrix(
            output,
            torch=self._torch,
            output_name="text",
            expected_batch_size=1,
        )[0]

    def _encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        inputs = self._processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with self._torch.inference_mode():
            output = self._model.get_image_features(
                pixel_values=pixel_values
            )
        return _feature_matrix(
            output,
            torch=self._torch,
            output_name="image",
            expected_batch_size=len(images),
        )

    def score_frames(
        self,
        *,
        video_path: Path,
        frames: Sequence[FrameRecord],
        images_by_id: Mapping[int, Image.Image],
        query: str,
    ) -> dict[int, float]:
        if not isinstance(query, str):
            raise TypeError("query must be a non-empty string")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must be a non-empty string")

        ordered_frames = tuple(frames)
        frame_ids = tuple(frame.frame_id for frame in ordered_frames)
        if any(type(frame_id) is not int for frame_id in frame_ids):
            raise ValueError("frame_ids must contain exact integers")
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("frame_ids must be unique")
        for frame_id in frame_ids:
            if frame_id not in images_by_id:
                raise ValueError(f"Missing image for frame_id {frame_id}")
        if not ordered_frames:
            return {}

        features: list[np.ndarray | None] = [None] * len(ordered_frames)
        misses: list[tuple[int, FrameRecord]] = []
        for index, frame in enumerate(ordered_frames):
            cached = self._cache.load(video_path, frame)
            if cached is None:
                misses.append((index, frame))
            else:
                features[index] = cached

        try:
            for start in range(0, len(misses), self._image_batch_size):
                batch = misses[start : start + self._image_batch_size]
                batch_images = [
                    images_by_id[frame.frame_id] for _, frame in batch
                ]
                encoded = self._encode_images(batch_images)
                for (index, frame), feature in zip(
                    batch, encoded, strict=True
                ):
                    self._cache.store(video_path, frame, feature)
                    features[index] = feature
            query_feature = self._encode_text(normalized_query)
        except self._torch.cuda.OutOfMemoryError as exc:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            raise RuntimeError(
                f"CLIP scoring ran out of memory for {len(ordered_frames)} frames"
            ) from exc

        ordered_features = [
            feature for feature in features if feature is not None
        ]
        if len(ordered_features) != len(ordered_frames):
            raise RuntimeError("missing CLIP frame feature after encoding")
        dimensions = {feature.shape for feature in ordered_features}
        if len(dimensions) != 1:
            raise ValueError("frame feature dimensions must match")
        frame_feature_matrix = np.stack(ordered_features)
        return cosine_similarity_scores(
            frame_ids, frame_feature_matrix, query_feature
        )


def _numeric_array(value: np.ndarray, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    try:
        converted = array.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must convert safely to float64") from exc
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values")
    return converted


def cosine_similarity_scores(
    frame_ids: Sequence[int],
    frame_features: np.ndarray,
    query_feature: np.ndarray,
) -> dict[int, float]:
    ids = tuple(frame_ids)
    if any(type(frame_id) is not int for frame_id in ids):
        raise ValueError("frame_ids must contain exact integers")
    if len(set(ids)) != len(ids):
        raise ValueError("frame_ids must be unique")

    frames = _numeric_array(frame_features, name="frame_features")
    query = _numeric_array(query_feature, name="query_feature")
    if frames.ndim != 2:
        raise ValueError("frame_features must be exactly two-dimensional")
    if query.ndim != 1:
        raise ValueError("query_feature must be exactly one-dimensional")
    if frames.shape[0] != len(ids):
        raise ValueError("frame_ids length must match frame_features rows")
    if frames.shape[1] <= 0 or query.size <= 0:
        raise ValueError("features must have a positive feature dimension")
    if frames.shape[1] != query.shape[0]:
        raise ValueError("frame and query feature dimensions must match")

    query_scale = np.max(np.abs(query))
    if query_scale == 0.0:
        raise ValueError("query feature has zero norm")
    scaled_query = query / query_scale
    normalized_query = scaled_query / np.linalg.norm(scaled_query)

    if not ids:
        return {}

    frame_scales = np.max(np.abs(frames), axis=1)
    zero_rows = np.flatnonzero(frame_scales == 0.0)
    if zero_rows.size:
        raise ValueError(
            f"frame feature has zero norm at row {int(zero_rows[0])}"
        )
    scaled_frames = frames / frame_scales[:, None]
    normalized_frames = scaled_frames / np.linalg.norm(
        scaled_frames, axis=1
    )[:, None]
    similarities = np.clip(normalized_frames @ normalized_query, -1.0, 1.0)
    return {
        frame_id: float(similarity)
        for frame_id, similarity in zip(ids, similarities, strict=True)
    }
