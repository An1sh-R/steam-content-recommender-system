"""HTTP endpoints. Routes select and serialize; they never compute."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_connection, get_engine
from api.schemas import GameOption, GameSummary, Recommendation
from recommender import catalogue, config

router = APIRouter()


@router.get("/health")
def health(con=Depends(get_connection)) -> dict:
    games = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    return {"status": "ok", "games": games}


@router.get("/games", response_model=list[GameOption])
def search_games(
    q: str = Query("", description="substring of the game title"),
    limit: int = Query(20, ge=1, le=100),
    con=Depends(get_connection),
) -> list[GameOption]:
    """Options for the search-and-select widget."""
    rows = catalogue.search_names(con, q, limit)
    return [GameOption(appid=row["appid"], label=_label(row)) for row in rows]


@router.get("/games/{appid}", response_model=GameSummary)
def get_game(appid: int, con=Depends(get_connection)) -> GameSummary:
    rows = catalogue.get_games(con, [appid])
    if not rows:
        raise HTTPException(status_code=404, detail=f"No game with AppID {appid}")
    return GameSummary.from_row(rows[0])


@router.get("/popular", response_model=list[GameSummary])
def popular(
    k: int = Query(24, ge=1, le=100),
    con=Depends(get_connection),
) -> list[GameSummary]:
    """The landing page: quality-aware popularity, no query needed."""
    return [GameSummary.from_row(row) for row in catalogue.browse(con, limit=k)]


@router.get("/recommend/{appid}", response_model=list[Recommendation])
def recommend(
    appid: int,
    k: int = Query(12, ge=1, le=50),
    diversity: float = Query(config.DEFAULT_DIVERSITY, ge=0.0, le=1.0),
    engine=Depends(get_engine),
) -> list[Recommendation]:
    """Games similar to ``appid``. The primary workflow.

    ``diversity`` is the MMR trade-off: 0 is pure similarity, 1 ignores it.
    """
    if not engine.knows(appid):
        raise HTTPException(status_code=404, detail=f"No game with AppID {appid}")
    rows = engine.similar(appid, k=k, diversity=diversity)
    return [Recommendation.from_row(row) for row in rows]


def _label(row: dict) -> str:
    """Disambiguate same-named games in the picker."""
    parts = [row["name"]]
    if row["release_year"]:
        parts.append(f"({row['release_year']})")
    if row["developers"]:
        parts.append(f"- {row['developers'].split(',')[0]}")
    return " ".join(parts)
