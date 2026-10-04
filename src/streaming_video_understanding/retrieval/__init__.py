"""Feature retrieval and caching utilities."""

from .base import OptionAwareFrameScorer, SemanticFrameScorer
from .cache import FrameFeatureCache
from .clip import ClipFrameScorer, cosine_similarity_scores
from .option import OptionScoreItem, aggregate_option_scores

__all__ = [
    "ClipFrameScorer",
    "FrameFeatureCache",
    "OptionAwareFrameScorer",
    "OptionScoreItem",
    "SemanticFrameScorer",
    "aggregate_option_scores",
    "cosine_similarity_scores",
]
