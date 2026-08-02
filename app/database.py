"""The SQLite catalogue: how we build it, and how we read it.

The database is rebuilt from scratch by app/build.py and is never written to
while the app is running. It holds game details only -- the TF-IDF matrices are
separate .npz files, because a big pile of numbers is not something SQL is good
at storing or searching.

Tags and genres get their own small tables so we can filter and count them.
"""

import sqlite3

import numpy as np

# Columns we sort the browse page by. Anything not in here is rejected, which is
# also what stops a caller injecting SQL through the sort_by parameter.
SORTABLE_COLUMNS = ["popularity", "total_reviews", "release_year", "price", "name"]

# The typeahead searches and sorts using only these columns, so SQLite can
# answer it from one index without ever opening the (very large) games table.
SEARCH_INDEX_COLUMNS = ["name", "popularity", "release_year", "developers", "appid"]

# The "+" in front of popularity looks like a typo but is doing real work: it
# stops SQLite using the popularity index, which would make it test every game's
# name one row at a time. See docs/ENGINEERING.md.
SEARCH_SQL = """
    SELECT appid, name, release_year, developers
    FROM games
    WHERE name LIKE ? COLLATE NOCASE
    ORDER BY +popularity DESC
    LIMIT ?
"""

# Glues the tag and genre tables back on, so reading a game gives us everything
# we need to draw its card in one query.
TAG_AND_GENRE_COLUMNS = """
    (SELECT group_concat(tag, ',')   FROM game_tags   WHERE appid = games.appid) AS tags,
    (SELECT group_concat(genre, ',') FROM game_genres WHERE appid = games.appid) AS genres
"""


def build_database(games, path):
    """Write the cleaned catalogue to a fresh SQLite file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    # tags/genres/categories hold Python lists, which SQLite cannot store.
    # Tags and genres go into their own tables below; categories are only used
    # when building the TF-IDF documents, so they are simply dropped here.
    main_table = games.drop(columns=["tags", "genres", "categories"])

    connection = sqlite3.connect(path)
    main_table.to_sql("games", connection, index=False)

    connection.execute("CREATE UNIQUE INDEX idx_games_appid ON games(appid)")
    for column in SORTABLE_COLUMNS:
        connection.execute(f"CREATE INDEX idx_games_{column} ON games({column})")
    connection.execute(
        f"CREATE INDEX idx_games_search ON games({', '.join(SEARCH_INDEX_COLUMNS)})"
    )

    # One row per (game, tag) and one per (game, genre).
    for column, table, value_column in [
        ("tags", "game_tags", "tag"),
        ("genres", "game_genres", "genre"),
    ]:
        pairs = games[["appid", column]].explode(column).dropna()
        pairs = pairs.rename(columns={column: value_column})
        pairs.to_sql(table, connection, index=False)
        connection.execute(f"CREATE INDEX idx_{table}_value ON {table}({value_column})")
        connection.execute(f"CREATE INDEX idx_{table}_appid ON {table}(appid)")

    connection.commit()
    connection.close()


def connect(path):
    """Open the catalogue. Rows come back as dict-like objects."""
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def get_games(connection, appids):
    """Look up full details for these AppIDs, keeping the order they were asked for.

    SQL gives no ordering guarantee for `IN`, and the order matters here because
    the caller has already ranked these games. AppIDs we do not have are simply
    left out, so never assume the result is the same length as the input.
    """
    if not appids:
        return []

    placeholders = ",".join("?" * len(appids))
    sql = f"""
        SELECT games.*, {TAG_AND_GENRE_COLUMNS}
        FROM games
        WHERE appid IN ({placeholders})
    """

    games_by_appid = {}
    for row in connection.execute(sql, appids):
        games_by_appid[row["appid"]] = dict(row)

    ordered = []
    for appid in appids:
        if appid in games_by_appid:
            ordered.append(games_by_appid[appid])
    return ordered


def browse_games(connection, name="", genres=None, max_price=None,
                 sort_by="popularity", limit=24):
    """The browse page. With no filters this is just the most popular games.

    Passing several genres narrows the results: a game must have all of them.
    """
    if sort_by not in SORTABLE_COLUMNS:
        raise ValueError(f"Cannot sort by {sort_by!r}. Try one of {SORTABLE_COLUMNS}.")

    conditions = []
    values = []

    if name:
        conditions.append("name LIKE ? COLLATE NOCASE")
        values.append(f"%{name}%")

    if max_price is not None:
        conditions.append("price <= ?")
        values.append(max_price)

    for genre in genres or []:
        conditions.append("appid IN (SELECT appid FROM game_genres WHERE genre = ?)")
        values.append(genre)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    sql = f"SELECT appid FROM games {where_clause} ORDER BY {sort_by} DESC LIMIT ?"
    values.append(limit)

    appids = [row["appid"] for row in connection.execute(sql, values)]
    return get_games(connection, appids)


def search_games(connection, query, limit=20):
    """Find games whose title contains `query`, best known games first.

    Returns AppIDs as well as names because plenty of games share a title.
    """
    rows = connection.execute(SEARCH_SQL, (f"%{query}%", limit)).fetchall()
    return [dict(row) for row in rows]


def count_tag_usage(connection):
    """How many games carry each tag, most common tag first.

    Explanations use these counts: a rare shared tag says much more about two
    games than a common one does.
    """
    sql = "SELECT tag, COUNT(*) FROM game_tags GROUP BY tag ORDER BY 2 DESC"
    counts = {}
    for tag, count in connection.execute(sql):
        counts[tag] = count
    return counts


def list_genres(connection):
    """Every genre in the catalogue, most common first. Used by the filter menu."""
    sql = "SELECT genre FROM game_genres GROUP BY genre ORDER BY COUNT(*) DESC"
    return [row["genre"] for row in connection.execute(sql)]


def load_popularity(connection, appids):
    """Popularity scores lined up with the rows of the TF-IDF matrices.

    Reranking needs one popularity value per matrix row, and the matrices are
    ordered by `appids`, so we read the scores back in that same order.
    """
    stored = {}
    for appid, popularity in connection.execute("SELECT appid, popularity FROM games"):
        stored[appid] = popularity

    scores = []
    for appid in appids:
        scores.append(stored.get(int(appid), 0.0))
    return np.array(scores, dtype=float)
