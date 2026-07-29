"""SQLite catalogue: build it, and query it for browsing and hydration.

The DB is a read-only derived artifact -- rebuilt from scratch by
``recommender.build``, never written to at request time. SQL earns its place
here because faceted filtering (genre AND tag AND platform AND price) is
exactly what indexed relational queries are good at.

Vectors do *not* live here; they are .npz files loaded into memory once.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from recommender import schema

# Multi-value fields are exploded into indexed child tables so they can be filtered on.
CHILD_TABLES = {
    "tags": ("game_tags", "tag"),
    "genres": ("game_genres", "genre"),
    "categories": ("game_categories", "category"),
}

PLATFORMS = {"windows", "mac", "linux"}
SORT_COLUMNS = {"total_reviews", "release_year", "price", "name", "popularity"}

# Exactly what the typeahead selects and orders by, so its scan is served by an
# index and never touches the games table -- which carries the descriptions and
# is ~180 MB. See search_names.
SEARCH_COLUMNS = ("name", "popularity", "release_year", "developers", "appid")

# Module level so the test can assert on its query plan without restating it.
SEARCH_SQL = """
    SELECT appid, name, release_year, developers
    FROM games
    WHERE name LIKE ? COLLATE NOCASE
    ORDER BY +popularity DESC
    LIMIT ?
"""

# Re-joins each child table into a comma-separated column, so reads return
# display-ready rows without the caller issuing extra queries.
_LIST_COLUMNS = """
    (SELECT group_concat(tag, ',')      FROM game_tags       WHERE appid = g.appid) AS tags,
    (SELECT group_concat(genre, ',')    FROM game_genres     WHERE appid = g.appid) AS genres,
    (SELECT group_concat(category, ',') FROM game_categories WHERE appid = g.appid) AS categories
"""


def build_db(df: pd.DataFrame, path: Path) -> None:
    """Write the cleaned catalogue to SQLite, replacing any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    games = df.drop(columns=list(schema.MULTIVALUE_COLUMNS))

    with sqlite3.connect(path) as con:
        games.to_sql("games", con, index=False)
        con.execute("CREATE UNIQUE INDEX idx_games_appid ON games(appid)")

        # Browse always ORDERs BY one of these; without an index SQLite sorts
        # the whole catalogue before applying LIMIT. Guarded by what exists so
        # new sort columns (popularity, M2) are picked up automatically.
        for column in SORT_COLUMNS & set(games.columns):
            con.execute(f"CREATE INDEX idx_games_{column} ON games({column})")

        if set(SEARCH_COLUMNS) <= set(games.columns):
            covered = ", ".join(SEARCH_COLUMNS)
            con.execute(f"CREATE INDEX idx_games_search ON games({covered})")

        for column, (table, value_column) in CHILD_TABLES.items():
            pairs = df[["appid", column]].explode(column).dropna()
            pairs.rename(columns={column: value_column}).to_sql(table, con, index=False)
            con.execute(f"CREATE INDEX idx_{table}_value ON {table}({value_column})")
            con.execute(f"CREATE INDEX idx_{table}_appid ON {table}(appid)")


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def get_games(con: sqlite3.Connection, appids: list[int]) -> list[dict]:
    """Hydrate display-ready metadata for the given AppIDs, in the order passed in.

    Every read path funnels through here: select the AppIDs you want first, then
    hydrate just those. Attaching the tag/genre lists before narrowing makes
    SQLite build them for the whole catalogue (~100ms) rather than for a page (~1ms).
    """
    if not appids:
        return []

    placeholders = ",".join("?" * len(appids))
    sql = f"SELECT g.*, {_LIST_COLUMNS} FROM games g WHERE g.appid IN ({placeholders})"
    by_id = {row["appid"]: dict(row) for row in con.execute(sql, appids)}
    return [by_id[appid] for appid in appids if appid in by_id]


def browse(
    con: sqlite3.Connection,
    genres: list[str] | None = None,
    tags: list[str] | None = None,
    categories: list[str] | None = None,
    platform: str | None = None,
    max_price: float | None = None,
    name: str | None = None,
    sort_by: str = "popularity",
    limit: int = 60,
) -> list[dict]:
    """Faceted browse. Multiple genres/tags narrow the results (AND semantics)."""
    if sort_by not in SORT_COLUMNS:
        raise ValueError(f"sort_by must be one of {sorted(SORT_COLUMNS)}")

    clauses: list[str] = []
    params: list = []

    if name:
        clauses.append("g.name LIKE ? COLLATE NOCASE")
        params.append(f"%{name}%")

    if max_price is not None:
        clauses.append("g.price <= ?")
        params.append(max_price)

    if platform:
        if platform not in PLATFORMS:
            raise ValueError(f"platform must be one of {sorted(PLATFORMS)}")
        clauses.append(f"g.{platform} = 1")

    for values, (table, value_column) in (
        (genres or [], CHILD_TABLES["genres"]),
        (tags or [], CHILD_TABLES["tags"]),
        (categories or [], CHILD_TABLES["categories"]),
    ):
        for value in values:
            clauses.append(f"g.appid IN (SELECT appid FROM {table} WHERE {value_column} = ?)")
            params.append(value)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT appid FROM games g {where} ORDER BY g.{sort_by} DESC LIMIT ?"
    appids = [row[0] for row in con.execute(sql, [*params, limit])]
    return get_games(con, appids)


def search_names(con: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    """Substring name search for the select widget, best-known games first.

    Returns AppIDs, never titles alone -- 1,210 games share a name with another.

    The ``+`` on ``popularity`` is load-bearing, not a typo. Without it SQLite
    satisfies the ORDER BY by walking idx_games_popularity and testing LIKE row
    by row until it has ``limit`` matches. That is fast for a common substring
    and catastrophic for a specific one: 0.1 ms for "a" but 1,070 ms for a title
    that matches nothing -- and typing a specific title is what a typeahead is
    for. The unary plus makes the ordering non-indexable, so SQLite scans
    idx_games_search once and sorts the handful of matches: a flat ~6 ms.
    """
    rows = con.execute(SEARCH_SQL, (f"%{query}%", limit)).fetchall()
    return [dict(row) for row in rows]


def value_counts(con: sqlite3.Connection, column: str) -> dict[str, int]:
    """How many games carry each value, most common first.

    Browse uses the keys as facet options; explanations use the counts, because
    the rarest shared tag is the one that says something.
    """
    table, value_column = CHILD_TABLES[column]
    sql = f"SELECT {value_column}, COUNT(*) FROM {table} GROUP BY {value_column} ORDER BY 2 DESC"
    return {row[0]: row[1] for row in con.execute(sql)}


def distinct_values(con: sqlite3.Connection, column: str) -> list[str]:
    """Facet options for the browse UI, most common first."""
    return list(value_counts(con, column))


def popularity_by_appid(con: sqlite3.Connection, appids) -> np.ndarray:
    """Popularity aligned to the given AppID order, i.e. to the TF-IDF rows.

    Reranking needs a popularity per matrix row. Reading it from the catalogue
    at startup keeps it out of the .npz artifacts, which stay purely vectors.
    """
    scores = dict(con.execute("SELECT appid, popularity FROM games"))
    return np.array([scores.get(int(appid), 0.0) for appid in appids], dtype=float)
