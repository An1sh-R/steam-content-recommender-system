"""Evaluation metrics.

Every metric here earns its place by answering a question that would change what
we do next:

    ndcg@10        Is the ranking better?              -- the headline
    recall@50      Is retrieval or ranking the limit?  -- sets N_CANDIDATES
    unique@10      Do we recycle the same few games?   -- detects a stuck model
    diversity@10   Are results near-duplicates?        -- regression guard
    novelty        Are we just showing blockbusters?   -- popularity bias
    poorly_rated   Are we recommending bad games?      -- what rerank is for
    tie_rate       Do scores actually discriminate?    -- V1 died of this
    same_publisher Have we regressed to V1?            -- integrity guard
    self_retrieval Does a game recommend itself?       -- must be 0

Two metrics were dropped for failing that bar:

    MAP@10       moves with NDCG@10; would never change a decision.
    Precision@10 needs an arbitrary relevance cut-off. At Jaccard >= 0.3 on
                 half-sized tag sets it reads ~0.08 for every model -- it
                 ordered them identically to NDCG@10 while looking alarming.
                 Graded NDCG says the same thing without the magic number.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def ndcg_at_k(ranked_relevance: np.ndarray, all_relevance: np.ndarray, k: int = 10) -> float:
    """Graded NDCG, ideal ordering taken over the whole catalogue."""
    gains = ranked_relevance[:k]
    ideal = np.sort(all_relevance)[::-1][:k]

    dcg = float((gains / np.log2(np.arange(2, len(gains) + 2))).sum())
    idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_rows: np.ndarray, all_relevance: np.ndarray, k: int = 50) -> float:
    """Share of the catalogue's 10 best matches that retrieval surfaced in top-k.

    Separates the two failure modes: low recall means retrieval never saw the
    good candidates, high recall with low NDCG means ranking is at fault.
    """
    ideal_top = set(np.argsort(all_relevance)[::-1][:10].tolist())
    return len(ideal_top & set(ranked_rows[:k].tolist())) / len(ideal_top)


def intra_list_diversity(
    ranked_rows: np.ndarray, tags: sparse.csr_matrix, k: int = 10
) -> float:
    """1 - mean pairwise tag cosine. Measured in the held-out tag space, which
    the model never saw, so a diverse-looking list cannot be an artifact."""
    rows = ranked_rows[:k]
    block = tags[rows]
    norms = np.sqrt(np.asarray(block.multiply(block).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0

    similarity = np.asarray((block @ block.T).todense()) / np.outer(norms, norms)
    upper = similarity[np.triu_indices(len(rows), k=1)]
    return float(1 - upper.mean()) if upper.size else 0.0


def novelty(ranked_rows: np.ndarray, popularity_percentile: np.ndarray, k: int = 10) -> float:
    """1 - mean popularity percentile. Higher means less blockbuster-biased."""
    return float(1 - popularity_percentile[ranked_rows[:k]].mean())


def poorly_rated_rate(
    ranked_rows: np.ndarray, review_ratio: np.ndarray, k: int = 10, threshold: float = 0.70
) -> float:
    """Share of the page holding games the community did not like.

    Added in M5. NDCG cannot see this by construction -- tag overlap is blind to
    whether a game is any good -- so without it the quality prior has no metric
    that can justify it, only an argument.
    """
    return float((review_ratio[ranked_rows[:k]] < threshold).mean())


def tie_rate(scores: np.ndarray, k: int = 50) -> float:
    """Share of the top-k that shares a score with another result.

    V1's quiz collapsed 112k games into 235 distinct vectors, leaving up to
    4,376 tied at maximum similarity, so popularity silently did the ranking.
    """
    top = scores[:k]
    return float(1 - len(np.unique(top)) / len(top)) if len(top) else 0.0


def same_publisher_rate(ranked_rows: np.ndarray, publishers: np.ndarray, query: int, k: int = 10):
    """V1 accidentally indexed publishers and became a publisher-matcher."""
    if not publishers[query]:
        return None
    return float((publishers[ranked_rows[:k]] == publishers[query]).mean())
