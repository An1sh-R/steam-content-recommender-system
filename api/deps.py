"""Shared, lazily-opened resources.

The SQLite connection is opened once and reused. No I/O happens at import time.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from recommender import catalogue, config


@lru_cache(maxsize=1)
def get_connection() -> sqlite3.Connection:
    if not config.CATALOGUE_DB.exists():
        raise RuntimeError(
            f"{config.CATALOGUE_DB} not found. Run: python -m recommender.build --sample"
        )
    return catalogue.connect(config.CATALOGUE_DB)
