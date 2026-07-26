"""The Streamlit page renders without raising.

Uses Streamlit's AppTest so this runs headlessly in CI, with the API stubbed
out -- the UI is presentation only, so it has no logic worth a live server.
"""

import pytest
import requests
from streamlit.testing.v1 import AppTest

from app import api_client
from recommender import config

FAKE_GAME = {
    "appid": 1,
    "name": "Test Game",
    "release_year": 2024,
    "price": 9.99,
    "header_image": "",
    "developers": "Test Studio",
    "tags": ["Roguelike", "Deck Building", "Indie"],
    "genres": ["Indie", "Strategy"],
    "positive": 950,
    "negative": 50,
    "total_reviews": 1000,
    "popularity": 0.87,
}


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(api_client, "health", lambda: {"status": "ok", "games": 56052})
    monkeypatch.setattr(api_client, "popular", lambda k=24: [FAKE_GAME] * 8)
    return AppTest.from_file(str(config.ROOT / "app" / "main.py"), default_timeout=30)


def test_landing_page_renders(app):
    app.run()
    assert not app.exception
    assert "Game Recommender" in app.title[0].value


def test_landing_page_shows_the_catalogue_size(app):
    app.run()
    assert any("56,052" in c.value for c in app.caption)


def test_landing_page_shows_games(app):
    app.run()
    rendered = " ".join(m.value for m in app.markdown)
    assert "Test Game" in rendered
    assert "store.steampowered.com/app/1" in rendered


def test_unreachable_api_shows_a_helpful_error(app, monkeypatch):
    def boom():
        raise requests.ConnectionError

    monkeypatch.setattr(api_client, "health", boom)
    app.run()

    assert not app.exception
    assert "Cannot reach the API" in app.error[0].value
