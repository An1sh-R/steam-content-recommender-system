"""Shared, lazily-opened resources.

The SQLite connection is opened once and reused. No I/O happens at import time.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from recommender import catalogue, config
from recommender.engine import Engine

_MISSING = "{path} not found. Run: python -m recommender.build --sample"


@lru_cache(maxsize=1)
def get_connection() -> sqlite3.Connection:
    if not config.CATALOGUE_DB.exists():
        raise RuntimeError(_MISSING.format(path=config.CATALOGUE_DB))
    return catalogue.connect(config.CATALOGUE_DB)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Artifacts are loaded once, on first use -- never at import time."""
    return Engine.load()
