"""Streamlit entry point."""

from __future__ import annotations

import requests
import streamlit as st

from app import api_client, components

st.set_page_config(page_title="Game Recommender", page_icon="🎮", layout="wide")


@st.cache_data(ttl=300)
def load_popular(k: int) -> list[dict]:
    return api_client.popular(k=k)


def main() -> None:
    st.title("🎮 Game Recommender")
    st.caption("Content-based Steam recommendations over a curated catalogue.")

    try:
        status = api_client.health()
    except requests.RequestException:
        st.error(
            "Cannot reach the API. Start it with:\n\n"
            "`uvicorn api.main:app --reload`\n\n"
            "and build the catalogue first: `python -m recommender.build --sample`"
        )
        return

    st.subheader("Popular right now")
    st.caption(
        f"Ranked by community approval (Wilson lower bound), reach and recency "
        f"across {status['games']:,} games."
    )
    components.game_grid(load_popular(24))


main()
