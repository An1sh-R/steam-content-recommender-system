"""Quality-aware reranking: community quality modifies relevance.
"""
from __future__ import annotations

import numpy as np

from recommender import config


def apply(similarity: np.ndarray, popularity: np.ndarray) -> np.ndarray:
    """Scale each similarity by its game's quality, in [floor, 1]."""
    floor = config.QUALITY_FLOOR
    return similarity * (floor + (1 - floor) * popularity)
