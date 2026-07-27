"""The Streamlit pages render without raising.

Uses Streamlit's AppTest so this runs headlessly in CI, with the API stubbed
out -- the UI is presentation only, so it has no logic worth a live server.
"""

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import api_client, components
from recommender import config

FAKE_GAME = {
    "appid": 1,
    "name": "Test Game",
    "release_year": 2024,
    "price": 9.99,
    "developers": "Test Studio",
    "publishers": "Test Studio",
    "tags": ["Roguelike", "Deck Building", "Indie"],
    "genres": ["Indie", "Strategy"],
    "positive": 950,
    "negative": 50,
    "total_reviews": 1000,
    "popularity": 0.87,
}
FAKE_RECOMMENDATION = {
    **FAKE_GAME,
    "appid": 2,
    "name": "Similar Game",
    "similarity": 0.42,
    "parts": {"tags": 0.5, "genres": 0.3, "description": 0.4},
    "reasons": ["Shares 9 tags including Deck Building and Roguelike"],
}


@pytest.fixture
def app(monkeypatch):
    # @st.cache_data lives in the process, not the AppTest, so without this a
    # cached response from an earlier test satisfies the next one's stub.
    st.cache_data.clear()

    monkeypatch.setattr(api_client, "health", lambda: {"status": "ok", "games": 55973})
    monkeypatch.setattr(api_client, "browse", lambda **kw: [FAKE_GAME] * 8)
    monkeypatch.setattr(api_client, "facets", lambda column: ["Indie", "Action"])
    monkeypatch.setattr(
        api_client,
        "search_games",
        lambda q, limit=20: [{"appid": 1, "label": "Test Game (2024) - Test Studio"}],
    )
    monkeypatch.setattr(api_client, "get_game", lambda appid: FAKE_GAME)
    monkeypatch.setattr(api_client, "recommend", lambda appid, k=10: [FAKE_RECOMMENDATION] * k)
    return AppTest.from_file(str(config.ROOT / "app" / "main.py"), default_timeout=30)


def _text(app) -> str:
    parts = [*app.markdown, *app.caption, *app.subheader]
    return " ".join(element.value for element in parts)


def test_browse_is_the_default_mode(app):
    app.run()
    assert not app.exception
    assert "Game Recommender" in app.title[0].value
    assert "Browse the catalogue" in _text(app)


def test_browse_shows_the_catalogue_size_and_games(app):
    app.run()
    rendered = _text(app)
    assert "55,973" in rendered
    assert "Test Game" in rendered
    assert "store.steampowered.com/app/1" in rendered


def test_switching_mode_shows_recommendations_with_reasons(app):
    app.run()
    app.sidebar.radio[0].set_value("Recommend similar games").run()

    assert not app.exception
    rendered = _text(app)
    assert "Similar Game" in rendered
    assert "Shares 9 tags including Deck Building and Roguelike" in rendered


def test_recommendation_count_is_selectable(app):
    app.run()
    app.sidebar.radio[0].set_value("Recommend similar games").run()

    assert app.radio[0].options == ["5", "10", "20"]
    app.radio[0].set_value(5).run()
    assert not app.exception


def test_empty_browse_results_explain_themselves(app, monkeypatch):
    monkeypatch.setattr(api_client, "browse", lambda **kw: [])
    app.run()
    assert "No games match those filters" in app.info[0].value


def test_unreachable_api_shows_a_helpful_error(app, monkeypatch):
    def boom():
        raise requests.ConnectionError

    monkeypatch.setattr(api_client, "health", boom)
    app.run()

    assert not app.exception
    assert "Cannot reach the API" in app.error[0].value


# --- components ----------------------------------------------------------


def test_artwork_is_derived_from_the_appid():
    """No image dataset, no asset pipeline, no column in the catalogue."""
    assert components.artwork({"appid": 620}).endswith("/steam/apps/620/header.jpg")


def test_byline_does_not_repeat_a_self_published_studio():
    assert components.byline(FAKE_GAME) == "Test Studio · 2024"
    assert components.byline({**FAKE_GAME, "publishers": "Big Co"}) == (
        "Test Studio / Big Co · 2024"
    )


def test_approval_handles_a_game_with_no_reviews():
    assert components.approval({"total_reviews": 0, "positive": 0}) == "No reviews"


def test_free_games_are_labelled_not_priced():
    assert components.price(0) == "Free"
    assert components.price(9.99) == "$9.99"
