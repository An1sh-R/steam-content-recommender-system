"""The HTTP endpoints.

The engine is swapped for one built from the sample, so these tests never touch
the real catalogue and run in seconds.
"""

import pytest
from fastapi.testclient import TestClient

from app import api


@pytest.fixture(scope="module")
def client(engine):
    api.get_engine.cache_clear()
    api.app.dependency_overrides[api.get_engine] = lambda: engine
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def a_popular_appid(client):
    return client.get("/browse", params={"limit": 1}).json()[0]["appid"]


def test_health_reports_the_catalogue_size(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["games"] > 0


def test_browse_returns_popular_games_as_json_ready_for_a_card(client):
    games = client.get("/browse", params={"limit": 10}).json()

    assert len(games) == 10
    assert [g["popularity"] for g in games] == sorted(
        [g["popularity"] for g in games], reverse=True
    )
    # Tags and genres are stored comma-joined but must come back as lists.
    assert isinstance(games[0]["tags"], list)
    assert isinstance(games[0]["genres"], list)


def test_browse_filters_by_genre_and_rejects_a_bad_sort_column(client):
    genre = client.get("/genres").json()[0]
    games = client.get("/browse", params={"genres": [genre], "limit": 5}).json()
    assert games and all(genre in game["genres"] for game in games)

    assert client.get("/browse", params={"sort_by": "; DROP TABLE"}).status_code == 422


def test_search_returns_appids_with_readable_labels(client, a_popular_appid):
    name = client.get(f"/games/{a_popular_appid}").json()["name"]
    results = client.get("/games", params={"q": name.lower(), "limit": 20}).json()

    assert any(name in result["label"] for result in results)
    assert all(isinstance(result["appid"], int) for result in results)
    assert client.get("/games", params={"q": "zzzznotagame"}).json() == []


def test_a_game_can_be_looked_up_and_unknown_ones_404(client, a_popular_appid):
    assert client.get(f"/games/{a_popular_appid}").json()["appid"] == a_popular_appid
    assert client.get("/games/-1").status_code == 404


def test_recommend_returns_k_distinct_games_and_never_the_query(client, a_popular_appid):
    results = client.get(f"/recommend/{a_popular_appid}", params={"k": 6}).json()

    assert len(results) == 6
    assert len({game["appid"] for game in results}) == 6
    assert all(game["appid"] != a_popular_appid for game in results)


def test_every_recommendation_explains_itself(client, a_popular_appid):
    results = client.get(f"/recommend/{a_popular_appid}").json()

    for game in results:
        assert game["reasons"]
        assert 0 <= game["similarity"] <= 1
        assert set(game["parts"]) == {"tags", "genres", "description"}


def test_out_of_range_and_unknown_requests_are_rejected(client, a_popular_appid):
    assert client.get("/recommend/-1").status_code == 404
    assert client.get(f"/recommend/{a_popular_appid}", params={"k": 0}).status_code == 422
    assert client.get(f"/recommend/{a_popular_appid}", params={"k": 999}).status_code == 422
    assert client.get("/browse", params={"limit": 0}).status_code == 422
