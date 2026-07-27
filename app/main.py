"""Streamlit entry point: two modes over the same API."""

from __future__ import annotations

import requests
import streamlit as st

from app import api_client, components

st.set_page_config(page_title="Game Recommender", page_icon="🎮", layout="wide")

RESULT_COUNTS = [5, 10, 20]
SORT_OPTIONS = {
    "Popularity": "popularity",
    "Review count": "total_reviews",
    "Newest": "release_year",
    "Price": "price",
    "Name": "name",
}


# Streamlit reruns the whole script on every widget interaction, so without
# caching each keystroke would re-issue every request behind the page.
@st.cache_data(ttl=300)
def load_browse(q, genres, max_price, sort_by, limit) -> list[dict]:
    return api_client.browse(
        q=q, genres=list(genres), max_price=max_price, sort_by=sort_by, limit=limit
    )


@st.cache_data(ttl=300)
def load_genres() -> list[str]:
    return api_client.facets("genres")


@st.cache_data(ttl=300)
def load_options(query: str) -> list[dict]:
    return api_client.search_games(query, limit=25)


@st.cache_data(ttl=300)
def load_game(appid: int) -> dict:
    return api_client.get_game(appid)


@st.cache_data(ttl=300)
def load_recommendations(appid: int, k: int) -> list[dict]:
    return api_client.recommend(appid, k=k)


def browse_page() -> None:
    st.subheader("Browse the catalogue")

    query = st.text_input("Search by title", placeholder="e.g. Hades")
    filters, sorting = st.columns([3, 1])
    genres = filters.multiselect("Genres", load_genres())
    sort_label = sorting.selectbox("Sort by", list(SORT_OPTIONS))
    free_only = st.checkbox("Free games only")

    games = load_browse(
        query,
        tuple(genres),  # hashable, so the cache key is stable
        0.0 if free_only else None,
        SORT_OPTIONS[sort_label],
        24,
    )

    if not games:
        st.info("No games match those filters. Try removing one.")
        return

    st.caption(f"Showing {len(games)} games.")
    components.game_grid(games)


def similar_page() -> None:
    st.subheader("Find games like one you love")

    query = st.text_input("Search for a game", placeholder="e.g. Stardew Valley")
    options = load_options(query)
    if not options:
        st.info("No games match that title.")
        return

    # Everything keys on AppID -- 1,210 games share a name with another game,
    # so the label is for humans and the AppID is what gets sent.
    choice = st.selectbox("Pick a game", options, format_func=lambda o: o["label"])
    count = st.radio("How many recommendations?", RESULT_COUNTS, index=1, horizontal=True)

    components.game_detail(load_game(choice["appid"]))
    st.divider()

    st.markdown("#### Recommended because you like this")
    components.game_grid(load_recommendations(choice["appid"], count))


def main() -> None:
    st.title("🎮 Game Recommender")

    try:
        status = api_client.health()
    except requests.RequestException:
        st.error(
            "Cannot reach the API. Start it with:\n\n"
            "`uvicorn api.main:app --reload`\n\n"
            "and build the catalogue first: `python -m recommender.build --sample`"
        )
        return

    st.caption(
        f"Content-based recommendations over {status['games']:,} Steam games, "
        "ranked by similarity and community quality."
    )

    mode = st.sidebar.radio("Mode", ["Browse", "Recommend similar games"])
    st.sidebar.caption(
        "Similarity comes from tags, genres and descriptions. Results are then "
        "scaled by a Wilson-based community quality score."
    )

    if mode == "Browse":
        browse_page()
    else:
        similar_page()


main()
