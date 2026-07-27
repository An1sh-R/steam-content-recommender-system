"""Maximal Marginal Relevance: greedy diversification of a ranked candidate list.

Content retrieval clusters. Ask for games like Portal 2 and the top of the list
is other puzzle-platformers that are also near-duplicates of *each other*, so the
list carries less information than its length suggests. MMR picks each slot by
score *minus* how much the candidate repeats what is already chosen.

    value(i) = (1 - d) * score(i) - d * max_{j in chosen} similarity(i, j)

``d = 0`` reduces exactly to plain top-k, so there is no separate code path for
the undiversified case.
"""

from __future__ import annotations

import numpy as np


def select(scores: np.ndarray, pairwise: np.ndarray, k: int, diversity: float) -> np.ndarray:
    """Indices into ``scores`` of the chosen candidates, best first."""
    k = min(k, len(scores))
    if k <= 0:
        return np.empty(0, dtype=int)

    chosen = [int(np.argmax(scores))]
    while len(chosen) < k:
        redundancy = pairwise[:, chosen].max(axis=1)
        value = (1 - diversity) * scores - diversity * redundancy
        value[chosen] = -np.inf
        chosen.append(int(np.argmax(value)))

    return np.asarray(chosen)
