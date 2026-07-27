"""Quality-aware reranking: community quality modifies relevance.

Multiplicative, not additive. V1 blended `0.6*sim + 0.2*pop + ...`, which lets a
popular-but-irrelevant game outrank a relevant one -- one term compensates for
another. A multiplier can only reorder *within* a band of similar relevance; it
can never promote something the query is unrelated to. The floor bounds how much
an unpopular game can be punished.
"""

from __future__ import annotations

import numpy as np

from recommender import config


def apply(similarity: np.ndarray, popularity: np.ndarray) -> np.ndarray:
    """Scale each similarity by its game's quality, in [floor, 1]."""
    floor = config.QUALITY_FLOOR
    return similarity * (floor + (1 - floor) * popularity)
