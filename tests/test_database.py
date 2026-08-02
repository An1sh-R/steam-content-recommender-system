"""The SQLite catalogue."""

import pytest

from app import database


def test_the_database_matches_the_dataframe_it_was_built_from(connection, games):
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"games", "game_tags", "game_genres"} <= tables

    game_count = connection.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    tag_count = connection.execute("SELECT COUNT(*) FROM game_tags").fetchone()[0]
    assert game_count == len(games)
    assert tag_count == games["tags"].map(len).sum()

    # Tags and genres live in their own tables, not as columns on games.
    columns = {row[1] for row in connection.execute("PRAGMA table_info(games)")}
    assert "tags" not in columns and "genres" not in columns


def test_get_games_returns_them_in_the_order_asked_for(connection, games):
    """Recommendations arrive already ranked, and SQL's IN does not preserve order."""
    appids = games["appid"].head(5).tolist()[::-1]
    rows = database.get_games(connection, appids)
    assert [row["appid"] for row in rows] == appids


def test_get_games_rejoins_tags_and_genres(connection, games):
    row = database.get_games(connection, [int(games["appid"].iloc[0])])[0]
    assert set(row["tags"].split(",")) == set(games["tags"].iloc[0])
    assert set(row["genres"].split(",")) == set(games["genres"].iloc[0])

    # Unknown AppIDs are dropped rather than raising, so callers must not index
    # the result by position.
    assert database.get_games(connection, []) == []
    assert database.get_games(connection, [-1]) == []


def test_browse_with_no_filters_is_the_popular_front_page(connection):
    """This is the cold-start answer: something good to show a brand new user."""
    games = database.browse_games(connection, limit=20)
    scores = [game["popularity"] for game in games]

    assert len(games) == 20
    assert scores == sorted(scores, reverse=True)


def test_browse_filters_narrow_the_results(connection):
    by_genre = database.browse_games(connection, genres=["Indie"], limit=1000)
    assert by_genre
    assert all("Indie" in game["genres"].split(",") for game in by_genre)

    free_games = database.browse_games(connection, max_price=0.0, limit=50)
    assert all(game["price"] == 0 for game in free_games)

    by_name = database.browse_games(connection, name="the", limit=20)
    assert all("the" in game["name"].lower() for game in by_name)


def test_browse_sorts_only_by_columns_it_recognises(connection):
    values = [
        game["total_reviews"]
        for game in database.browse_games(connection, sort_by="total_reviews", limit=20)
    ]
    assert values == sorted(values, reverse=True)

    # sort_by goes straight into the SQL, so it must be checked against a list.
    with pytest.raises(ValueError):
        database.browse_games(connection, sort_by="name; DROP TABLE games")


def test_search_uses_the_covering_index(connection):
    """Without this index SQLite tests every game's name one row at a time, which
    takes about a second for a title that matches nothing -- exactly what a
    search box gets typed into."""
    plan = [
        row[-1]
        for row in connection.execute(
            "EXPLAIN QUERY PLAN " + database.SEARCH_SQL, ("%x%", 20)
        )
    ]
    assert any("COVERING INDEX idx_games_search" in step for step in plan), plan
    assert not any("idx_games_popularity" in step for step in plan), plan


def test_search_finds_games_and_ranks_them_by_popularity(connection):
    """The trick that defeats the index must not change the answer."""
    results = database.search_games(connection, "a", limit=10)
    assert results
    assert all(isinstance(result["appid"], int) for result in results)

    full_rows = database.get_games(connection, [r["appid"] for r in results])
    scores = [row["popularity"] for row in full_rows]
    assert scores == sorted(scores, reverse=True)

    assert database.search_games(connection, "zzzznotagame") == []


def test_tag_counts_and_genres_are_ordered_by_how_common_they_are(connection):
    tag_counts = database.count_tag_usage(connection)
    counts = list(tag_counts.values())
    assert counts == sorted(counts, reverse=True)

    genres = database.list_genres(connection)
    assert "Indie" in genres
