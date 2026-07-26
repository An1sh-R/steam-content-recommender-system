"""Held-out tag protocol: ground truth the model never sees.

There are no user interactions in this dataset, so there is no behavioural
ground truth. Instead we evaluate the system as what it is -- an information
retrieval system -- using a held-out attribute split.

Each game's tags are split in half. The model under evaluation is fitted on one
half; relevance is judged on the other. Because the judging signal was never in
the feature space, the protocol is **not circular** -- which is the trap most
attribute-based proxies fall into.

This is a proxy for relevance, not human judgement. It ranks systems reliably;
it does not prove any single recommendation is good.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

MIN_TAGS = 8  # need enough tags that both halves are meaningful


def split_tags(tags: list[str]) -> tuple[list[str], list[str]]:
    """Deterministic alternating split: features get evens, evaluation gets odds."""
    return tags[0::2], tags[1::2]


def prepare(games: pd.DataFrame) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Return games with feature-half tags, plus a binary matrix of held-out tags."""
    halves = games["tags"].map(split_tags)

    evaluated = games.copy()
    evaluated["tags"] = [feature for feature, _ in halves]

    held_out = [held for _, held in halves]
    return evaluated, _binary_matrix(held_out)


def _binary_matrix(tag_lists: list[list[str]]) -> sparse.csr_matrix:
    vocabulary = {tag: i for i, tag in enumerate(sorted({t for tags in tag_lists for t in tags}))}
    rows, columns = [], []
    for row, tags in enumerate(tag_lists):
        for tag in tags:
            rows.append(row)
            columns.append(vocabulary[tag])
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, columns)), shape=(len(tag_lists), len(vocabulary))
    )


class Relevance:
    """Jaccard overlap of held-out tags, computed against the whole catalogue."""

    def __init__(self, held_out: sparse.csr_matrix) -> None:
        self.held_out = held_out
        self.sizes = np.asarray(held_out.sum(axis=1)).ravel()

    def against_all(self, row: int) -> np.ndarray:
        intersection = np.asarray((self.held_out @ self.held_out[row].T).todense()).ravel()
        union = self.sizes + self.sizes[row] - intersection
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def sample_queries(games: pd.DataFrame, n: int, seed: int = 0) -> np.ndarray:
    """Query rows stratified across popularity deciles.

    Without stratification the sample is dominated by the long tail, and the
    metrics stop describing the games anyone would actually search for.
    """
    eligible = games.index[games["tags"].map(len) >= MIN_TAGS]
    if len(eligible) <= n:
        return np.asarray(eligible)

    deciles = pd.qcut(games.loc[eligible, "popularity"], 10, labels=False, duplicates="drop")
    per_decile = max(1, n // (deciles.max() + 1))

    rng = np.random.default_rng(seed)
    picked = [
        rng.choice(group.index, size=min(per_decile, len(group)), replace=False)
        for _, group in deciles.groupby(deciles)
    ]
    return np.sort(np.concatenate(picked))
