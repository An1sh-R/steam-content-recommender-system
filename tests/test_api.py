import pytest
from fastapi.testclient import TestClient

from api import deps
from api.main import app
from recommender import catalogue


@pytest.fixture(scope="module")
def client(games, tmp_path_factory):
    path = tmp_path_factory.mktemp("api") / "catalogue.db"
    catalogue.build_db(games, path)

    deps.get_connection.cache_clear()
    app.dependency_overrides[deps.get_connection] = lambda: catalogue.connect(path)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["games"] > 0


def test_popular_returns_k_games_in_descending_order(client):
    games = client.get("/popular", params={"k": 10}).json()
    assert len(games) == 10
    scores = [g["popularity"] for g in games]
    assert scores == sorted(scores, reverse=True)


def test_popular_serialises_tags_as_a_list(client):
    game = client.get("/popular", params={"k": 1}).json()[0]
    assert isinstance(game["tags"], list)
    assert isinstance(game["genres"], list)


def test_popular_rejects_out_of_range_k(client):
    assert client.get("/popular", params={"k": 0}).status_code == 422
    assert client.get("/popular", params={"k": 999}).status_code == 422


def test_game_detail_round_trip(client):
    appid = client.get("/popular", params={"k": 1}).json()[0]["appid"]
    detail = client.get(f"/games/{appid}").json()
    assert detail["appid"] == appid


def test_unknown_game_returns_404(client):
    assert client.get("/games/-1").status_code == 404


def test_search_returns_appids_and_labels(client):
    options = client.get("/games", params={"q": "a", "limit": 5}).json()
    assert options
    assert all(isinstance(o["appid"], int) and o["label"] for o in options)


def test_search_matches_case_insensitively(client):
    name = client.get("/popular", params={"k": 1}).json()[0]["name"]
    hits = client.get("/games", params={"q": name.lower(), "limit": 20}).json()
    assert any(name in option["label"] for option in hits)


def test_search_with_no_match_is_empty(client):
    assert client.get("/games", params={"q": "zzzznotagame"}).json() == []
