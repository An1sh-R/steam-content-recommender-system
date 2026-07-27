import pytest
from fastapi.testclient import TestClient

from api import deps
from api.main import app
from recommender import catalogue, documents, vectorize
from recommender.engine import Engine


@pytest.fixture(scope="module")
def client(games, tmp_path_factory):
    path = tmp_path_factory.mktemp("api") / "catalogue.db"
    catalogue.build_db(games, path)
    connection = catalogue.connect(path)

    matrices = vectorize.fit(documents.build_documents(games))
    engine = Engine(matrices, games["appid"].to_numpy(), connection)

    deps.get_connection.cache_clear()
    app.dependency_overrides[deps.get_connection] = lambda: connection
    app.dependency_overrides[deps.get_engine] = lambda: engine
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


def test_recommend_returns_k_distinct_games(client):
    """Ordering is asserted in test_retrieval, where the ranking score is visible.

    The API deliberately exposes ``similarity`` rather than the final score, and
    similarity is *not* monotone down the page once the quality prior applies --
    that reordering is exactly what reranking is for.
    """
    appid = client.get("/popular", params={"k": 1}).json()[0]["appid"]
    recommendations = client.get(f"/recommend/{appid}", params={"k": 6}).json()

    assert len(recommendations) == 6
    assert len({r["appid"] for r in recommendations}) == 6


def test_recommend_explains_itself(client):
    appid = client.get("/popular", params={"k": 1}).json()[0]["appid"]
    recommendations = client.get(f"/recommend/{appid}").json()
    assert all(r["reasons"] for r in recommendations)


def test_recommend_excludes_the_query(client):
    appid = client.get("/popular", params={"k": 1}).json()[0]["appid"]
    recommendations = client.get(f"/recommend/{appid}").json()
    assert all(r["appid"] != appid for r in recommendations)


def test_recommend_exposes_the_score_breakdown(client):
    appid = client.get("/popular", params={"k": 1}).json()[0]["appid"]
    parts = client.get(f"/recommend/{appid}").json()[0]["parts"]
    assert set(parts) == {"tags", "genres", "description"}


def test_recommend_unknown_game_returns_404(client):
    assert client.get("/recommend/-1").status_code == 404


def test_recommend_rejects_out_of_range_k(client):
    appid = client.get("/popular", params={"k": 1}).json()[0]["appid"]
    assert client.get(f"/recommend/{appid}", params={"k": 0}).status_code == 422


def test_browse_with_no_filters_is_the_popular_front_page(client):
    games = client.get("/browse", params={"limit": 10}).json()
    assert len(games) == 10
    scores = [g["popularity"] for g in games]
    assert scores == sorted(scores, reverse=True)


def test_browse_filters_by_title(client):
    name = client.get("/popular", params={"k": 1}).json()[0]["name"]
    games = client.get("/browse", params={"q": name[:6].lower()}).json()
    assert games and all(name[:6].lower() in g["name"].lower() for g in games)


def test_browse_narrows_by_genre(client):
    genre = client.get("/facets/genres").json()[0]
    games = client.get("/browse", params={"genres": [genre], "limit": 5}).json()
    assert games and all(genre in g["genres"] for g in games)


def test_browse_rejects_an_unknown_sort_column(client):
    assert client.get("/browse", params={"sort_by": "; DROP TABLE"}).status_code == 422


def test_facets_are_ordered_by_frequency(client):
    genres = client.get("/facets/genres").json()
    assert "Indie" in genres
    assert client.get("/facets/nonsense").status_code == 404
