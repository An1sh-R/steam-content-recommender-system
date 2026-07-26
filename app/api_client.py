"""The only place the UI talks HTTP."""

from __future__ import annotations

import os

import requests

BASE_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 10


def _get(path: str, **params) -> list | dict:
    response = requests.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def health() -> dict:
    return _get("/health")


def popular(k: int = 24) -> list[dict]:
    return _get("/popular", k=k)


def search_games(query: str, limit: int = 20) -> list[dict]:
    return _get("/games", q=query, limit=limit)


def get_game(appid: int) -> dict:
    return _get(f"/games/{appid}")
