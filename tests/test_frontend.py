"""The Streamlit interface.

Streamlit's AppTest runs the whole page headlessly. We stub requests.get rather
than the app's own functions, so the real HTTP-calling code still runs and only
the network is faked.
"""

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

import frontend.streamlit_app as ui
from app import config

APP_FILE = str(config.ROOT / "frontend" / "streamlit_app.py")

A_GAME = {
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
A_RECOMMENDATION = dict(
    A_GAME,
    appid=2,
    name="Similar Game",
    similarity=0.42,
    parts={"tags": 0.5, "genres": 0.3, "description": 0.4},
    reasons=["Shares 9 tags including Deck Building and Roguelike"],
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def fake_api(browse_results=None):
    """Build a stand-in for requests.get that answers from fixed data."""
    if browse_results is None:
        browse_results = [A_GAME] * 8

    def get(url, params=None, timeout=None):
        path = url.replace(ui.API_URL, "")
        params = params or {}

        if path == "/health":
            return FakeResponse({"status": "ok", "games": 55973})
        if path == "/genres":
            return FakeResponse(["Indie", "Action"])
        if path == "/browse":
            return FakeResponse(browse_results)
        if path == "/games":
            return FakeResponse([{"appid": 1, "label": "Test Game (2024) - Test Studio"}])
        if path.startswith("/games/"):
            return FakeResponse(A_GAME)
        if path.startswith("/recommend/"):
            return FakeResponse([A_RECOMMENDATION] * params.get("k", 10))
        raise AssertionError(f"unexpected request to {path}")

    return get


@pytest.fixture
def app(monkeypatch):
    # @st.cache_data lives in the process, so without this a response cached by
    # one test would satisfy the next test's stub.
    st.cache_data.clear()
    monkeypatch.setattr(requests, "get", fake_api())
    return AppTest.from_file(APP_FILE, default_timeout=30)


def rendered_text(app):
    elements = [*app.markdown, *app.caption, *app.subheader]
    return " ".join(element.value for element in elements)


def test_the_browse_page_loads_and_shows_games(app):
    app.run()
    text = rendered_text(app)

    assert not app.exception
    assert "Game Recommender" in app.title[0].value
    assert "Browse the catalogue" in text
    assert "55,973" in text, "the catalogue size comes from /health"
    assert "Test Game" in text
    assert "store.steampowered.com/app/1" in text


def test_recommendations_show_their_reasons_and_similarity_score(app):
    app.run()
    app.sidebar.radio[0].set_value("Recommend similar games").run()
    text = rendered_text(app)

    assert not app.exception
    assert "Similar Game" in text
    assert "Shares 9 tags including Deck Building and Roguelike" in text
    assert "Similarity 0.42" in text


def test_the_number_of_recommendations_can_be_changed(app):
    app.run()
    app.sidebar.radio[0].set_value("Recommend similar games").run()

    assert app.radio[0].options == ["5", "10", "20"]
    app.radio[0].set_value(20).run()
    assert not app.exception


def test_an_unreachable_api_explains_how_to_start_it(monkeypatch):
    st.cache_data.clear()

    def refuse(url, params=None, timeout=None):
        raise requests.ConnectionError

    monkeypatch.setattr(requests, "get", refuse)
    app = AppTest.from_file(APP_FILE, default_timeout=30)
    app.run()

    assert not app.exception
    assert "Cannot reach the API" in app.error[0].value


def test_empty_results_say_so_rather_than_showing_nothing(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(requests, "get", fake_api(browse_results=[]))

    app = AppTest.from_file(APP_FILE, default_timeout=30)
    app.run()

    assert "No games match those filters" in app.info[0].value


def test_numbers_are_formatted_to_fit_on_a_card():
    assert ui.format_approval({"total_reviews": 0, "positive": 0}) == "No reviews"
    assert ui.format_approval({"total_reviews": 842, "positive": 800}) == "95% of 842 reviews"
    assert ui.format_approval(
        {"total_reviews": 1_150_098, "positive": 1_115_595}
    ) == "97% of 1.2M reviews"

    assert ui.format_price(0) == "Free"
    assert ui.format_price(9.99) == "$9.99"

    # Most indie games self-publish, so the studio should only appear once.
    assert ui.format_studios(A_GAME) == "Test Studio · 2024"
    assert ui.format_studios(dict(A_GAME, publishers="Big Co")) == (
        "Test Studio / Big Co · 2024"
    )
