"""The web interface. Two modes: browse the catalogue, or find similar games.

This file only draws things. Every number it shows comes from the API, which
means the interesting logic is all in app/, not here.

Run it with:  streamlit run frontend/streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 10

STEAM_PAGE = "https://store.steampowered.com/app/{appid}"

# Steam serves cover art at a predictable address for every game, so we can work
# it out from the AppID instead of storing thousands of images ourselves.
COVER_ART = "https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"

HOW_MANY_OPTIONS = [5, 10, 20]
SORT_OPTIONS = {
    "Popularity": "popularity",
    "Review count": "total_reviews",
    "Newest": "release_year",
    "Price": "price",
    "Name": "name",
}


# --- Talking to the API --------------------------------------------------
#
# Streamlit re-runs this whole file every time the user touches a widget, so
# without @st.cache_data every keystroke would fire off every request again.


def call_api(path, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def fetch_health():
    return call_api("/health")


@st.cache_data(ttl=300)
def fetch_genres():
    return call_api("/genres")


@st.cache_data(ttl=300)
def fetch_browse(name, genres, max_price, sort_by, limit):
    params = {"q": name, "genres": list(genres), "sort_by": sort_by, "limit": limit}
    if max_price is not None:
        params["max_price"] = max_price
    return call_api("/browse", **params)


@st.cache_data(ttl=300)
def fetch_search_results(query):
    return call_api("/games", q=query, limit=25)


@st.cache_data(ttl=300)
def fetch_game(appid):
    return call_api(f"/games/{appid}")


@st.cache_data(ttl=300)
def fetch_recommendations(appid, how_many):
    return call_api(f"/recommend/{appid}", k=how_many)


# --- Drawing games -------------------------------------------------------


def format_price(price):
    if price == 0:
        return "Free"
    return f"${price:.2f}"


def format_review_count(count):
    """1150098 -> "1.2M". Written out in full it wraps a card onto two lines."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M".replace(".0", "")
    if count >= 1_000:
        return f"{count / 1_000:.1f}K".replace(".0", "")
    return str(count)


def format_approval(game):
    total = game["total_reviews"]
    if not total:
        return "No reviews"
    percent = round(100 * game["positive"] / total)
    return f"{percent}% of {format_review_count(total)} reviews"


def format_studios(game):
    """Developer and publisher, then the year. For most indie games these are the
    same studio, so we only print it once."""
    studios = []
    for studio in [game.get("developers"), game.get("publishers")]:
        if studio and studio not in studios:
            studios.append(studio)

    parts = []
    if studios:
        parts.append(" / ".join(studios))
    if game.get("release_year"):
        parts.append(str(game["release_year"]))
    return " · ".join(parts)


def draw_game_card(game):
    """One game tile: cover art, title, rating, tags, and why it is here."""
    with st.container(border=True):
        # Old and delisted games have no cover art. Streamlit draws a placeholder
        # rather than crashing, which leaves us with a text-only card. Fine.
        st.image(COVER_ART.format(appid=game["appid"]), width="stretch")

        st.markdown(f"**[{game['name']}]({STEAM_PAGE.format(appid=game['appid'])})**")

        left, right = st.columns(2)
        left.caption(format_approval(game))
        right.caption(format_price(game["price"]))

        if game.get("tags"):
            st.caption(" · ".join(game["tags"][:3]))

        # Only recommendations carry a similarity score.
        if game.get("similarity") is not None:
            st.caption(f"Similarity {game['similarity']:.2f}")

        for reason in game.get("reasons", []):
            st.caption(f"↳ {reason}")


def draw_game_grid(games, columns=4):
    for start in range(0, len(games), columns):
        row_of_games = games[start : start + columns]
        streamlit_columns = st.columns(columns)
        for column, game in zip(streamlit_columns, row_of_games):
            with column:
                draw_game_card(game)


def draw_game_details(game):
    """The game the user picked, shown large above its recommendations."""
    art_column, facts_column = st.columns([1, 2])

    with art_column:
        st.image(COVER_ART.format(appid=game["appid"]), width="stretch")

    with facts_column:
        st.subheader(game["name"])
        st.caption(format_studios(game))

        approval = "n/a"
        if game["total_reviews"]:
            approval = f"{round(100 * game['positive'] / game['total_reviews'])}%"

        stats = st.columns(3)
        stats[0].metric("Approval", approval)
        stats[1].metric("Reviews", f"{game['total_reviews']:,}")
        stats[2].metric("Popularity", f"{game['popularity']:.2f}")

        st.caption(format_price(game["price"]))
        if game.get("genres"):
            st.write(" ".join(f"`{genre}`" for genre in game["genres"]))
        if game.get("tags"):
            st.write(" ".join(f"`{tag}`" for tag in game["tags"][:10]))


# --- The two pages -------------------------------------------------------


def browse_page():
    st.subheader("Browse the catalogue")

    title = st.text_input("Search by title", placeholder="e.g. Hades")

    filter_column, sort_column = st.columns([3, 1])
    chosen_genres = filter_column.multiselect("Genres", fetch_genres())
    sort_label = sort_column.selectbox("Sort by", list(SORT_OPTIONS))
    free_only = st.checkbox("Free games only")

    max_price = 0.0 if free_only else None

    games = fetch_browse(
        title,
        tuple(chosen_genres),  # a tuple can be cached; a list cannot
        max_price,
        SORT_OPTIONS[sort_label],
        24,
    )

    if not games:
        st.info("No games match those filters. Try removing one.")
        return

    st.caption(f"Showing {len(games)} games.")
    draw_game_grid(games)


def recommend_page():
    st.subheader("Find games like one you love")

    title = st.text_input("Search for a game", placeholder="e.g. Stardew Valley")

    search_results = fetch_search_results(title)
    if not search_results:
        st.info("No games match that title.")
        return

    # Over a thousand games share a title with another game, so the dropdown
    # shows a readable label but everything we send uses the AppID.
    chosen = st.selectbox(
        "Pick a game", search_results, format_func=lambda result: result["label"]
    )
    how_many = st.radio(
        "How many recommendations?", HOW_MANY_OPTIONS, index=1, horizontal=True
    )

    draw_game_details(fetch_game(chosen["appid"]))
    st.divider()

    st.markdown("#### Recommended because you like this")
    draw_game_grid(fetch_recommendations(chosen["appid"], how_many))


def main():
    st.set_page_config(page_title="Game Recommender", page_icon="🎮", layout="wide")
    st.title("🎮 Game Recommender")

    try:
        health = fetch_health()
    except requests.RequestException:
        st.error(
            "Cannot reach the API. Start it with:\n\n"
            "`uvicorn app.api:app --reload`\n\n"
            "and build the catalogue first: `python -m app.build --sample`"
        )
        return

    st.caption(
        f"Content-based recommendations over {health['games']:,} Steam games, "
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
        recommend_page()


# Streamlit runs this file as a script, so this is what starts the app. The
# guard keeps `import frontend.streamlit_app` from drawing a page in the tests.
if __name__ == "__main__":
    main()
