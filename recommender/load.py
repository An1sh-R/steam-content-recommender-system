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

    # Pandas reads the rest of the columns correctly on its own: AppID and the
    # review counts come out as integers, Price as a float, and the platform
    # flags as booleans. Only these two need help.

    # Dates arrive as plain text, and clean.py reads .dt.year off this column.
    # format="mixed" because the file has both "Oct 21, 2008" and "Oct 2008".
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", format="mixed")

    # Missing text arrives as NaN, and NaN != "" is True -- so without this the
    # has-tags filter in clean.py would let all 42,502 untagged games through.
    for col in ["name", "description", "developers", "publishers", *MULTIVALUE_COLUMNS]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df
