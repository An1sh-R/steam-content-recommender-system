"""Quality-aware popularity score.

Steam reviews are binary thumbs up/down, so the right estimator for "how well
liked is this game" is the **Wilson score interval lower bound** -- not a mean
rating, and not a star-rating formula like IMDb's weighted rating, which
assumes a continuous scale and needs an arbitrary prior mean.

Wilson answers: *given these votes, what approval rate can we be 95% confident
the game is at least at?* Small samples are penalised automatically, so a
1-review game at 100% scores 0.21 while A Short Hike (18,904/19,064) scores
0.99. No hand-tuned minimum-review threshold is required.

Wilson alone is not enough for a landing page. It saturates above ~10k reviews,
so a 99%-of-10k indie would outrank a 96%-of-1M classic. Reach and recency are
blended in so the front page reflects popularity as well as approval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender import config


def wilson_lower_bound(
    positive: pd.Series | np.ndarray,
    negative: pd.Series | np.ndarray,
    z: float = config.WILSON_Z,
) -> np.ndarray:
    """Lower bound of the Wilson score interval for a binomial proportion."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    n = positive + negative

    # n == 0 has no evidence either way; score it 0 rather than dividing by zero.
    safe_n = np.where(n > 0, n, 1.0)
    p_hat = positive / safe_n

    denominator = 1 + z**2 / safe_n
    centre = p_hat + z**2 / (2 * safe_n)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / safe_n + z**2 / (4 * safe_n**2))

    return np.where(n > 0, np.clip((centre - margin) / denominator, 0.0, 1.0), 0.0)


def popularity_score(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.Series:
    """Blend approval, reach and freshness into a single [0, 1] ranking score.

    quality  Wilson lower bound -- confidence-adjusted approval
    reach    log-scaled review volume; counts span 1 to 8.8M, so log not linear
    recency  exponential decay on release date, mild by design
    """
    reference_date = reference_date or pd.Timestamp.now()
    weights = config.POPULARITY_WEIGHTS

    quality = wilson_lower_bound(df["positive"], df["negative"])

    reviews = (df["positive"] + df["negative"]).to_numpy(dtype=float)
    reach = np.log1p(reviews) / np.log1p(max(reviews.max(), 1.0))

    age_years = (reference_date - df["release_date"]).dt.days.to_numpy(dtype=float) / 365.25
    age_years = np.nan_to_num(age_years, nan=0.0).clip(min=0.0)  # unreleased -> treat as new
    recency = 0.5 ** (age_years / config.RECENCY_HALF_LIFE_YEARS)

    score = weights["quality"] * quality + weights["reach"] * reach + weights["recency"] * recency
    return pd.Series(score, index=df.index, name="popularity")
