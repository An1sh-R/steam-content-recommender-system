"""Catalogue filter and field normalisation.

Reduces the raw dump to games that can actually be recommended, and turns the
comma-joined string fields into lists.
"""

from __future__ import annotations

import pandas as pd

from recommender import config, load


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to the recommendable catalogue and normalise fields."""
    df = df[_is_recommendable(df)].copy()

    for col in load.MULTIVALUE_COLUMNS:
        df[col] = df[col].map(split_values)

    df["total_reviews"] = df["positive"] + df["negative"]
    df["release_year"] = df["release_date"].dt.year.astype("Int64")

    return _drop_reissues(df).reset_index(drop=True)


def _drop_reissues(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the same game listed under several AppIDs, keeping the main one.

    Steam carries legacy and regional SKUs -- Portal 2, Assassin's Creed 2 and
    BRINK each appear two or three times. They are perfect content matches, so without 
    this they rank first against themselves.

    Matching on name *and* description *and* developer is deliberately strict:
    349 rows share only a name and are genuinely different games.
    """
    return (
        df.sort_values("total_reviews", ascending=False)
        .drop_duplicates(subset=["name", "description", "developers"])
        .sort_index()
    )


def _is_recommendable(df: pd.DataFrame) -> pd.Series:
    """Games we can build a useful recommendation from.

    The review threshold is doing more work than it looks: games with zero
    reviews have 0.9% tag coverage, while games with any reviews have 100%.
    It filters unrecommendable entries, not merely unpopular ones.
    """
    return (
        (df["positive"] + df["negative"] >= config.MIN_REVIEWS)
        & (df["tags"].str.strip() != "")
        & (df["description"].str.split().map(len) >= config.MIN_DESCRIPTION_WORDS)
        & (~df["name"].str.contains(config.EXCLUDE_NAME_PATTERN, regex=True, na=False))
    )


def split_values(value: str) -> list[str]:
    """Split a comma-joined field into a unique, order-preserving list."""
    seen = {}
    for part in str(value).split(","):
        part = part.strip()
        if part:
            seen[part] = None
    return list(seen)
