"""Turns each game into one document per field group.
Not concatenating them into one document to prevent description from
dominating which is significantly larger than rest
"""

from __future__ import annotations

import re

import pandas as pd

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def build_documents(games: pd.DataFrame) -> dict[str, pd.Series]:
    """One text document per field group, keyed by space name."""
    return {
        "tags": games["tags"].map(_terms),
        # Genres are coarse but reliable; categories ("Single-player", "Co-op")
        # describe how a game is played. Both are small controlled vocabularies.
        "genres": (games["genres"] + games["categories"]).map(_terms),
        "description": games["description"].str.lower(),
    }


def _terms(values: list[str]) -> str:
    """Join multi-word terms into single tokens.

    "Turn-Based Strategy" becomes "turn_based_strategy" so it stays one term
    rather than three; underscores are word characters, so the default
    scikit-learn tokenizer keeps them intact.
    """
    return " ".join(_NON_ALPHANUMERIC.sub("_", value.lower()).strip("_") for value in values)
