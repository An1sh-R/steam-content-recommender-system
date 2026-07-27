"""Column contract for the raw Steam dataset.

THE HEADER DEFECT
-----------------
The published CSV has a malformed header row. It declares **39** column names,
but every data row contains **40** fields.

The culprit is header field 7, written as ``DiscountDLC count`` -- a missing
comma between ``Discount`` and ``DLC count``.

Because pandas aligns the 39 declared names against the first 39 fields, every
column from index 7 onward is labelled with its *neighbour's* name:

    header says ...          row actually holds ...
    "About the game"     ->  DLC count (an integer)
    "Positive"           ->  User score (0-100)
    "Categories"         ->  Publishers
    "Genres"             ->  Categories
    "Tags"               ->  Genres
    (no header)          ->  Movies

This is not cosmetic. A previous version of this project built its TF-IDF index
from the mislabelled columns and therefore indexed *Categories + Publishers +
Genres* while believing it indexed tags and descriptions. The resulting system
was effectively a publisher-matcher.

THE FIX
-------
Ignore the published header entirely and supply our own names positionally.
``RAW_COLUMNS`` below is the true 40-field ordering. ``validate_raw_shape``
fails loudly if a future dataset refresh changes the layout.
"""

from __future__ import annotations

import csv
from pathlib import Path

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

# The header the publisher actually ships. Kept so the defect is testable.
PUBLISHED_HEADER_FIELD_COUNT = 39
MERGED_HEADER_FIELD = "DiscountDLC count"

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


class RawSchemaError(RuntimeError):
    """Raised when the raw CSV no longer matches the documented layout."""


def validate_raw_shape(path: Path, sample_rows: int = 50) -> None:
    """Check the raw file still has the layout ``RAW_COLUMNS`` assumes.

    Guards against a dataset refresh silently changing the field order, which
    would corrupt every downstream feature without raising anything.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RawSchemaError(f"{path} is empty") from exc

        widths = {len(row) for _, row in zip(range(sample_rows), reader, strict=False)}

    if len(header) != PUBLISHED_HEADER_FIELD_COUNT:
        raise RawSchemaError(
            f"Expected the malformed {PUBLISHED_HEADER_FIELD_COUNT}-field header, "
            f"got {len(header)} fields. The upstream dataset layout has changed; "
            f"re-verify RAW_COLUMNS before trusting any feature."
        )
    if widths != {len(RAW_COLUMNS)}:
        raise RawSchemaError(
            f"Expected {len(RAW_COLUMNS)} fields per data row, saw widths {sorted(widths)}."
        )
