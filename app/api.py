"""The web API. Every endpoint the Streamlit front end calls lives here.

Endpoints look things up and hand them back. They never do any of the actual
recommending -- that is all in app/recommender.py.

Run it with:  uvicorn app.api:app --reload
"""

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from app import config, database, recommender

app = FastAPI(
    title="Game Recommender API",
    description="Content-based Steam game recommendations.",
    version="2.0.0",
)


# --- What the API sends back ---------------------------------------------


class GameSummary(BaseModel):
    """A game, with everything needed to draw its card."""

    appid: int
    name: str
    release_year: int | None = None
    price: float
    developers: str
    publishers: str
    tags: list[str]
    genres: list[str]
    positive: int
    negative: int
    total_reviews: int
    popularity: float


class Recommendation(GameSummary):
    """A recommended game, plus why we recommended it.

    `reasons` is what the user reads. `similarity` and `parts` are the numbers
    behind those reasons -- we send them too, because a recommendation you
    cannot check is not really explainable.
    """

    reasons: list[str]
    similarity: float
    parts: dict[str, float]


class SearchResult(BaseModel):
    """One line in the search box dropdown."""

    appid: int
    label: str


def to_game_summary(game):
    """Turn a database row into a GameSummary."""
    return GameSummary(
        appid=game["appid"],
        name=game["name"],
        release_year=game["release_year"],
        price=game["price"],
        developers=game["developers"],
        publishers=game["publishers"],
        tags=split_commas(game["tags"]),
        genres=split_commas(game["genres"]),
        positive=game["positive"],
        negative=game["negative"],
        total_reviews=game["total_reviews"],
        popularity=game["popularity"],
    )


def to_recommendation(game):
    """Turn a recommended game into a Recommendation."""
    summary = to_game_summary(game)
    return Recommendation(
        **summary.model_dump(),
        reasons=game["reasons"],
        similarity=game["similarity"],
        parts=game["parts"],
    )


def split_commas(value):
    """The database stores tags and genres joined by commas."""
    if not value:
        return []
    return value.split(",")


def search_label(game):
    """"Portal 2 (2011) - Valve". Plenty of games share a title, so the year and
    studio are what let the user tell them apart."""
    label = game["name"]
    if game["release_year"]:
        label += f" ({game['release_year']})"
    if game["developers"]:
        label += f" - {game['developers'].split(',')[0]}"
    return label


# --- The engine ----------------------------------------------------------


@lru_cache(maxsize=1)
def get_engine():
    """Load the engine once, the first time it is needed.

    lru_cache means every request after the first reuses the same one. Loading
    at import time instead would make the app impossible to test and would slow
    down every `uvicorn --reload`.
    """
    if not config.DATABASE.exists():
        raise RuntimeError(
            f"{config.DATABASE} not found. Build it: python -m app.build --sample"
        )
    return recommender.Engine.load()


# --- Endpoints -----------------------------------------------------------


@app.get("/health")
def health(engine=Depends(get_engine)):
    """Is the API up, and how many games does it know about?"""
    game_count = engine.connection.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    return {"status": "ok", "games": game_count}


@app.get("/games", response_model=list[SearchResult])
def search(
    q: str = Query("", description="part of a game title"),
    limit: int = Query(20, ge=1, le=100),
    engine=Depends(get_engine),
):
    """Search box suggestions."""
    matches = database.search_games(engine.connection, q, limit)
    return [SearchResult(appid=game["appid"], label=search_label(game)) for game in matches]


@app.get("/games/{appid}", response_model=GameSummary)
def get_game(appid: int, engine=Depends(get_engine)):
    """Everything we know about one game."""
    matches = database.get_games(engine.connection, [appid])
    if not matches:
        raise HTTPException(status_code=404, detail=f"No game with AppID {appid}")
    return to_game_summary(matches[0])


@app.get("/browse", response_model=list[GameSummary])
def browse(
    q: str = Query("", description="part of a game title"),
    genres: list[str] = Query(default_factory=list),
    max_price: float | None = Query(None, ge=0),
    sort_by: str = Query("popularity"),
    limit: int = Query(24, ge=1, le=60),
    engine=Depends(get_engine),
):
    """Browse the catalogue.

    With no filters this returns the most popular games, which is what the home
    page shows before the user has told us anything about their taste.
    """
    try:
        games = database.browse_games(
            engine.connection,
            name=q,
            genres=genres,
            max_price=max_price,
            sort_by=sort_by,
            limit=limit,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return [to_game_summary(game) for game in games]


@app.get("/genres", response_model=list[str])
def genres(engine=Depends(get_engine)):
    """Genre filter options, most common first."""
    return database.list_genres(engine.connection)


@app.get("/recommend/{appid}", response_model=list[Recommendation])
def recommend(
    appid: int,
    k: int = Query(12, ge=1, le=50),
    engine=Depends(get_engine),
):
    """Games similar to this one. This is what the whole project is for."""
    if not engine.knows(appid):
        raise HTTPException(status_code=404, detail=f"No game with AppID {appid}")
    games = recommender.recommend(engine, appid, count=k)
    return [to_recommendation(game) for game in games]
