"""Reusable Streamlit pieces. Presentation only -- no computation."""

from __future__ import annotations

import streamlit as st

STEAM_URL = "https://store.steampowered.com/app/{appid}"

# Artwork is derived from the AppID rather than stored. Steam serves a header
# image at a predictable path for every app, so the project needs no image
# dataset, no asset pipeline and no column in the catalogue for it.
ARTWORK_URL = "https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"


def artwork(game: dict) -> str:
    return ARTWORK_URL.format(appid=game["appid"])


def game_card(game: dict) -> None:
    """One game tile: art, title, approval, tags, and why it was recommended."""
    with st.container(border=True):
        # Delisted and very old titles have no header image. Streamlit renders a
        # broken-image placeholder rather than raising, so the card degrades to
        # its text -- which is the graceful fallback we want.
        st.image(artwork(game), width="stretch")

        st.markdown(f"**[{game['name']}]({STEAM_URL.format(appid=game['appid'])})**")

        left, right = st.columns(2)
        left.caption(approval(game))
        right.caption(price(game["price"]))

        if game.get("tags"):
            st.caption(" · ".join(game["tags"][:3]))

        for reason in game.get("reasons", []):
            st.caption(f"↳ {reason}")


def game_grid(games: list[dict], columns: int = 4) -> None:
    for row_start in range(0, len(games), columns):
        row = games[row_start : row_start + columns]
        # strict=False: the final row is usually shorter than the column count.
        for column, game in zip(st.columns(columns), row, strict=False):
            with column:
                game_card(game)


def game_detail(game: dict) -> None:
    """The selected game, shown larger than a card before its recommendations."""
    art, facts = st.columns([1, 2])
    with art:
        st.image(artwork(game), width="stretch")

    with facts:
        st.subheader(game["name"])
        st.caption(byline(game))

        stats = st.columns(3)
        stats[0].metric("Approval", _ratio(game))
        stats[1].metric("Reviews", f"{game['total_reviews']:,}")
        stats[2].metric("Popularity", f"{game['popularity']:.2f}")

        st.caption(price(game["price"]))
        if game.get("genres"):
            st.write(" ".join(f"`{genre}`" for genre in game["genres"]))
        if game.get("tags"):
            st.write(" ".join(f"`{tag}`" for tag in game["tags"][:10]))


def approval(game: dict) -> str:
    total = game["total_reviews"]
    if not total:
        return "No reviews"
    return f"{round(100 * game['positive'] / total)}% of {_compact(total)} reviews"


def _compact(count: int) -> str:
    """1150098 -> '1.2M'. Spelled out, a card's caption wraps onto two lines."""
    for limit, suffix in ((1_000_000, "M"), (1_000, "K")):
        if count >= limit:
            return f"{count / limit:.1f}{suffix}".replace(".0", "")
    return str(count)


def price(value: float) -> str:
    return "Free" if value == 0 else f"${value:.2f}"


def byline(game: dict) -> str:
    """Developer, publisher and year -- deduplicated, because for most indie
    games on Steam the developer and the publisher are the same studio."""
    studios = dict.fromkeys(filter(None, [game.get("developers"), game.get("publishers")]))
    parts = [" / ".join(studios)] if studios else []
    if game.get("release_year"):
        parts.append(str(game["release_year"]))
    return " · ".join(parts)


def _ratio(game: dict) -> str:
    total = game["total_reviews"]
    return f"{round(100 * game['positive'] / total)}%" if total else "n/a"
