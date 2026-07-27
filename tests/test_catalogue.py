import pytest

from recommender import catalogue


@pytest.fixture(scope="session")
def con(games, tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "catalogue.db"
    catalogue.build_db(games, path)
    return catalogue.connect(path)


def test_tables_exist(con):
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"games", "game_tags", "game_genres", "game_categories"} <= names


def test_row_counts_match_the_dataframe(con, games):
    assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == len(games)
    assert (
        con.execute("SELECT COUNT(*) FROM game_tags").fetchone()[0] == games["tags"].map(len).sum()
    )


def test_multivalue_columns_are_not_duplicated_on_games(con):
    columns = {r[1] for r in con.execute("PRAGMA table_info(games)")}
    assert "tags" not in columns and "genres" not in columns


def test_get_games_preserves_request_order(con, games):
    appids = games["appid"].head(5).tolist()[::-1]
    rows = catalogue.get_games(con, appids)
    assert [r["appid"] for r in rows] == appids


def test_get_games_returns_joined_lists(con, games):
    row = catalogue.get_games(con, [int(games["appid"].iloc[0])])[0]
    assert set(row["tags"].split(",")) == set(games["tags"].iloc[0])
    assert set(row["genres"].split(",")) == set(games["genres"].iloc[0])


def test_get_games_handles_empty_and_unknown(con):
    assert catalogue.get_games(con, []) == []
    assert catalogue.get_games(con, [-1]) == []


def test_browse_respects_limit(con):
    assert len(catalogue.browse(con, limit=7)) == 7


def test_browse_filters_by_genre(con):
    rows = catalogue.browse(con, genres=["Indie"], limit=50)
    assert rows
    assert all("Indie" in r["genres"].split(",") for r in rows)


def test_browse_narrows_with_multiple_facets(con):
    one = catalogue.browse(con, genres=["Indie"], limit=1000)
    two = catalogue.browse(con, genres=["Indie"], tags=["Singleplayer"], limit=1000)
    assert len(two) <= len(one)
    assert all("Singleplayer" in r["tags"].split(",") for r in two)


def test_browse_filters_by_price_and_platform(con):
    rows = catalogue.browse(con, max_price=0.0, platform="linux", limit=50)
    assert all(r["price"] == 0 and r["linux"] == 1 for r in rows)


def test_browse_sorts_descending(con):
    values = [r["total_reviews"] for r in catalogue.browse(con, sort_by="total_reviews", limit=20)]
    assert values == sorted(values, reverse=True)


def test_browse_defaults_to_popularity(con):
    values = [r["popularity"] for r in catalogue.browse(con, limit=20)]
    assert values == sorted(values, reverse=True)


def test_browse_rejects_unknown_sort_and_platform(con):
    with pytest.raises(ValueError):
        catalogue.browse(con, sort_by="; DROP TABLE games")
    with pytest.raises(ValueError):
        catalogue.browse(con, platform="switch")


def test_distinct_values_is_ordered_by_frequency(con):
    genres = catalogue.distinct_values(con, "genres")
    assert genres
    counts = [
        con.execute("SELECT COUNT(*) FROM game_genres WHERE genre = ?", (g,)).fetchone()[0]
        for g in genres[:5]
    ]
    assert counts == sorted(counts, reverse=True)


def test_name_search_is_served_by_the_covering_index(con):
    """The typeahead must not walk idx_games_popularity testing LIKE per row.

    That plan is 0.1 ms for "a" and 1,070 ms for a title matching nothing -- and
    typing a specific title is exactly what the widget is for. See search_names.
    """
    plan = [row[-1] for row in con.execute(
        "EXPLAIN QUERY PLAN " + catalogue.SEARCH_SQL, ("%x%", 20)
    )]

    assert any("COVERING INDEX idx_games_search" in step for step in plan), plan
    assert not any("idx_games_popularity" in step for step in plan), plan


def test_name_search_still_ranks_by_popularity(con):
    """The + that defeats the index must not change the answer."""
    rows = catalogue.search_names(con, "a", limit=10)
    by_appid = {r["appid"]: r for r in catalogue.get_games(con, [r["appid"] for r in rows])}
    scores = [by_appid[r["appid"]]["popularity"] for r in rows]
    assert scores == sorted(scores, reverse=True)
