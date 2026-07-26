"""Reusable Streamlit pieces. Presentation only -- no computation."""

from __future__ import annotations

import streamlit as st

STEAM_URL = "https://store.steampowered.com/app/{appid}"


def game_card(game: dict) -> None:
    """One game tile: art, title, approval, tags."""
    with st.container(border=True):
        if game.get("header_image"):
            st.image(game["header_image"], use_container_width=True)

        st.markdown(f"**[{game['name']}]({STEAM_URL.format(appid=game['appid'])})**")

        left, right = st.columns(2)
        left.caption(_approval(game))
        right.caption(_price(game["price"]))

        if game["tags"]:
            st.caption(" · ".join(game["tags"][:3]))


def game_grid(games: list[dict], columns: int = 4) -> None:
    for row_start in range(0, len(games), columns):
        row = games[row_start : row_start + columns]
        # strict=False: the final row is usually shorter than the column count.
        for column, game in zip(st.columns(columns), row, strict=False):
            with column:
                game_card(game)


def _approval(game: dict) -> str:
    total = game["total_reviews"]
    if not total:
        return "No reviews"
    return f"{round(100 * game['positive'] / total)}% of {total:,} reviews"


def _price(price: float) -> str:
    return "Free" if price == 0 else f"${price:.2f}"
