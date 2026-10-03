from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence

from PIL import Image

from ..data.schema import FrameRecord


class SemanticFrameScorer(Protocol):
    def score_frames(
        self,
        *,
        video_path: Path,
        frames: Sequence[FrameRecord],
        images_by_id: Mapping[int, Image.Image],
        query: str,
    ) -> dict[int, float]: ...
