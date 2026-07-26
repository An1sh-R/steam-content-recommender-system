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

# Re-joins each child table into a comma-separated column, so reads return
# display-ready rows without the caller issuing extra queries.
_LIST_COLUMNS = ", ".join(
    f"(SELECT group_concat({value_column}, ',') FROM {table} WHERE appid = g.appid) AS {column}"
    for column, (table, value_column) in CHILD_TABLES.items()
)


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
    sort_by: str = "total_reviews",
    limit: int = 60,
) -> list[dict]:
    """Faceted browse. Multiple genres/tags narrow the results (AND semantics)."""
    if sort_by not in SORT_COLUMNS:
        raise ValueError(f"sort_by must be one of {sorted(SORT_COLUMNS)}")

    clauses: list[str] = []
    params: list = []

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


def distinct_values(con: sqlite3.Connection, column: str) -> list[str]:
    """Facet options for the browse UI, most common first."""
    table, value_column = CHILD_TABLES[column]
    sql = f"SELECT {value_column} FROM {table} GROUP BY {value_column} ORDER BY COUNT(*) DESC"
    return [row[0] for row in con.execute(sql).fetchall()]
