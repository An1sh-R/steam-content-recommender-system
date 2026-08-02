"""Score breakdown -> short human reasons.

Every line here restates a number the pipeline already computed. Nothing is
inferred about a game beyond what ranked it: a field is named only when it
carried a real share of the combined score, so the reasons follow the ranking
instead of describing the game in general.

Returns plain strings, but short ones -- the UI decides layout, not wording.
"""

from __future__ import annotations

from recommender import config

_PHRASES = {
    "tags": "Similar tags",  # fallback; the tag line below is almost always used
    "genres": "Similar genres",
    "description": "Similar gameplay description",
}


def reasons(query: dict, candidate: dict, parts: dict[str, float], tag_counts: dict[str, int]):
    """Why ``candidate`` was recommended for ``query``. At most a few lines."""
    lines = []

    for field in _leading_fields(parts):
        shared = shared_tags(query, candidate, tag_counts) if field == "tags" else []
        lines.append(_tag_line(shared) if shared else _PHRASES[field])

    if _highly_rated(candidate):
        lines.append("Highly rated by the community")

    developer = _shared_developer(query, candidate)
    if developer:
        lines.append(f"Also by {developer}")

    return lines


def shared_tags(query: dict, candidate: dict, tag_counts: dict[str, int]) -> list[str]:
    """Tags both games carry, rarest first.

    Rarity is informativeness -- the same reasoning as the IDF that weighted
    these tags during retrieval. "Both are Indie" explains nothing; "both are
    Deck Building Roguelikes" explains the recommendation.
    """
    shared = set(_values(query, "tags")) & set(_values(candidate, "tags"))
    return sorted(shared, key=lambda tag: tag_counts.get(tag, 0))


def _leading_fields(parts: dict[str, float]) -> list[str]:
    """The spaces that actually drove the score, largest contribution first.

    A share of the total, not an absolute threshold: cosines are not comparable
    across three vocabularies of 449, 86 and 30,000 terms, but their weighted
    contributions to one score are.
    """
    contribution = {
        field: config.FIELD_WEIGHTS.get(field, 0.0) * value for field, value in parts.items()
    }
    total = sum(contribution.values())
    if total <= 0:
        return []

    ranked = sorted(contribution, key=contribution.get, reverse=True) # type: ignore
    return [f for f in ranked if contribution[f] / total >= config.EXPLAIN_MIN_SHARE]


def _tag_line(shared: list[str]) -> str:
    named = " and ".join(shared[:2])
    if len(shared) <= 2:
        return f"Shares {named}"
    return f"Shares {len(shared)} tags including {named}"


def _highly_rated(game: dict) -> bool:
    """Deliberately the raw review ratio, not the popularity score.

    The claim is about ratings, so it is backed by ratings. `popularity` also
    folds in reach and recency, which would make the sentence untrue.
    """
    reviews = game.get("total_reviews") or 0
    if reviews < config.ACCLAIM_MIN_REVIEWS:
        return False
    return game["positive"] / reviews >= config.ACCLAIM_MIN_RATIO


def _shared_developer(query: dict, candidate: dict) -> str | None:
    """Developers are excluded from similarity (they dominated V1) but they are a
    legitimate *explanation* once a game has earned its place on content."""
    shared = set(_values(query, "developers")) & set(_values(candidate, "developers"))
    return next(iter(sorted(shared)), None)


def _values(game: dict, field: str) -> list[str]:
    """Catalogue rows carry multi-value fields as comma-joined strings."""
    value = game.get(field)
    if not value:
        return []
    return value.split(",") if isinstance(value, str) else list(value)
