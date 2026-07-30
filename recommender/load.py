"""Raw CSV -> typed DataFrame.

THE HEADER DEFECT
-----------------
The published CSV has a malformed header row. It declares **39** column names,
but every data row contains **40** fields. The culprit is header field 7,
written as ``DiscountDLC count`` -- a missing comma between ``Discount`` and
``DLC count``.

Because pandas aligns the 39 declared names against the first 39 fields, every
column from index 7 onward is labelled with its *neighbour's* name:

    header says ...          row actually holds ...
    "About the game"     ->  DLC count (an integer)
    "Positive"           ->  User score (0-100)
    "Categories"         ->  Publishers
    "Genres"             ->  Categories
    "Tags"               ->  Genres
    (no header)          ->  Movies

This is not cosmetic: a TF-IDF index built from the mislabelled columns indexes
*Categories + Publishers + Genres* while believing it indexes tags and
descriptions. So we ignore the published header entirely and supply our own
names positionally. ``RAW_COLUMNS`` is the true 40-field ordering, verified
against the data rather than the header.

Loading is deliberately separate from cleaning: this module only fixes the
column contract and coerces types. Row filtering happens in ``clean.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recommender import config

# True field order of the raw CSV, as verified against the data (not the header).
RAW_COLUMNS = [
    "AppID",
    "Name",
    "Release date",
    "Estimated owners",
    "Peak CCU",
    "Required age",
    "Price",
    "Discount",  # header merges this with the next column
    "DLC count",
    "About the game",
    "Supported languages",
    "Full audio languages",
    "Reviews",
    "Header image",
    "Website",
    "Support url",
    "Support email",
    "Windows",
    "Mac",
    "Linux",
    "Metacritic score",
    "Metacritic url",
    "User score",
    "Positive",
    "Negative",
    "Score rank",
    "Achievements",
    "Recommendations",
    "Notes",
    "Average playtime forever",
    "Average playtime two weeks",
    "Median playtime forever",
    "Median playtime two weeks",
    "Developers",
    "Publishers",
    "Categories",
    "Genres",
    "Tags",
    "Screenshots",
    "Movies",
]

# Only these are read into memory. A column earns its place by feeding
# recommendation, browsing, or explanations -- nothing else is carried.
#   Recommendations : redundant with Positive + Negative
#   Achievements    : no bearing on similarity, quality or browsing
#   Metacritic      : only 3.4% of games have a score
USED_COLUMNS = [
    "AppID",
    "Name",
    "Release date",
    "Estimated owners",
    "Price",
    "About the game",
    "Windows",
    "Mac",
    "Linux",
    "Positive",
    "Negative",
    "Developers",
    "Publishers",
    "Categories",
    "Genres",
    "Tags",
]

# Raw CSV names -> the snake_case names used everywhere downstream.
COLUMN_RENAME = {
    "AppID": "appid",
    "Name": "name",
    "Release date": "release_date",
    "Estimated owners": "estimated_owners",
    "Price": "price",
    "About the game": "description",
    "Windows": "windows",
    "Mac": "mac",
    "Linux": "linux",
    "Positive": "positive",
    "Negative": "negative",
    "Developers": "developers",
    "Publishers": "publishers",
    "Categories": "categories",
    "Genres": "genres",
    "Tags": "tags",
}

NUMERIC_COLUMNS = ["price", "positive", "negative"]
BOOLEAN_COLUMNS = ["windows", "mac", "linux"]
MULTIVALUE_COLUMNS = ["categories", "genres", "tags"]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw dataset with correctly aligned columns.

    ``header=0`` skips the malformed published header; ``names=RAW_COLUMNS``
    supplies the true 40-field layout positionally. See the module docstring.
    """
    path = Path(path or config.RAW_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the Steam dataset there, or pass the "
            f"committed sample at {config.SAMPLE_CSV}."
        )

    df = pd.read_csv(
        path,
        header=0,
        names=RAW_COLUMNS,
        usecols=USED_COLUMNS,
        index_col=False,
        low_memory=False,
    )
    df = df.rename(columns=COLUMN_RENAME)
    return _coerce_types(df)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df["appid"] = pd.to_numeric(df["appid"], errors="coerce").astype("Int64")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in BOOLEAN_COLUMNS:
        df[col] = df[col].astype(str).str.strip().str.lower().eq("true")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", format="mixed")

    for col in ["name", "description", "developers", "publishers"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in MULTIVALUE_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df
