"""Candidate retrieval: weighted cosine similarity across the TF-IDF spaces."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from recommender import config


def similar_rows(
    matrices: dict[str, sparse.csr_matrix],
    row: int,
    n: int = config.N_CANDIDATES,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Top-``n`` most similar rows to ``row``.

    Returns the row indices, their combined scores, and the per-space scores
    behind them -- the breakdown is what explanations are built from.
    """
    weights = weights or config.FIELD_WEIGHTS

    per_space = {name: _cosine_against_all(matrices[name], row) for name in weights}
    combined = sum(weight * per_space[name] for name, weight in weights.items())

    # Exclude the query itself by position
    combined[row] = -np.inf # type: ignore

    top = _top_indices(combined, n) # type: ignore
    return top, combined[top], {name: scores[top] for name, scores in per_space.items()} # type: ignore


def _cosine_against_all(matrix: sparse.csr_matrix, row: int) -> np.ndarray:
    """Rows are L2-normalised, so the dot product is already the cosine.

    Written as ``matrix @ query.T`` rather than ``query @ matrix.T``: the latter
    transposes the full 56k x 30k matrix on every call, which measured ~1.5x
    slower for identical results.
    """
    return np.asarray((matrix @ matrix[row].T).todense()).ravel()


def _top_indices(scores: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(scores))
    candidates = np.argpartition(-scores, n - 1)[:n]
    return candidates[np.argsort(-scores[candidates])]
