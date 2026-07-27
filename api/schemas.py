"""Response models. The API's public shape lives here and nowhere else."""

from __future__ import annotations

from pydantic import BaseModel


class GameSummary(BaseModel):
    """A game as rendered on a card."""

    appid: int
    name: str
    release_year: int | None = None
    price: float
    header_image: str
    developers: str
    tags: list[str]
    genres: list[str]
    positive: int
    negative: int
    total_reviews: int
    popularity: float

    @classmethod
    def from_row(cls, row: dict) -> GameSummary:
        return cls(
            **{key: row[key] for key in cls.model_fields if key not in {"tags", "genres"}},
            tags=_split(row.get("tags")),
            genres=_split(row.get("genres")),
        )


class Recommendation(GameSummary):
    """A recommended game, with why it is here.

    ``reasons`` is what the UI shows. ``similarity`` and ``parts`` (the
    per-space cosines) are the numbers behind those reasons -- kept because the
    score breakdown is the point of having three spaces, and because a
    recommendation you cannot audit is not explainable.
    """

    reasons: list[str]
    similarity: float
    parts: dict[str, float]


class GameOption(BaseModel):
    """One entry in the search-and-select widget."""

    appid: int
    label: str


def _split(value: str | None) -> list[str]:
    return value.split(",") if value else []
