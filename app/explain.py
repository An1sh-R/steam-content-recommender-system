"""Turns a score breakdown into a few short sentences the user can read.

Every reason here restates something the algorithm actually computed. We never
say a game is similar because of its tags unless tags really did drive its
score, so the explanation always matches the ranking.
"""

from app import config

FIELD_WEIGHTS = {
    "tags": config.TAG_WEIGHT,
    "genres": config.GENRE_WEIGHT,
    "description": config.DESCRIPTION_WEIGHT,
}

PLAIN_ENGLISH = {
    "tags": "Similar tags",
    "genres": "Similar genres",
    "description": "Similar gameplay description",
}


def explain_recommendation(query_game, recommended_game, score_parts, tag_usage_counts):
    """Why was `recommended_game` suggested for `query_game`? A few short lines."""
    reasons = []

    # Name whichever of the three models actually drove the score.
    for field in leading_fields(score_parts):
        shared = []
        if field == "tags":
            shared = shared_tags(query_game, recommended_game, tag_usage_counts)

        if not shared:
            reasons.append(PLAIN_ENGLISH[field])
        elif len(shared) <= 2:
            reasons.append(f"Shares {' and '.join(shared)}")
        else:
            named = " and ".join(shared[:2])
            reasons.append(f"Shares {len(shared)} tags including {named}")

    # We use the raw review ratio here, not the popularity score, because the
    # sentence is a claim about ratings. Popularity also folds in how many
    # people played it and how new it is, which would make the claim untrue.
    total_reviews = recommended_game.get("total_reviews") or 0
    if total_reviews >= config.ACCLAIM_MIN_REVIEWS:
        positive_ratio = recommended_game["positive"] / total_reviews
        if positive_ratio >= config.ACCLAIM_MIN_RATIO:
            reasons.append("Highly rated by the community")

    # Developers are deliberately left out of the similarity models, but once a
    # game has earned its place a shared studio is still worth mentioning.
    shared_developers = set(split_field(query_game, "developers")) & set(
        split_field(recommended_game, "developers")
    )
    if shared_developers:
        reasons.append(f"Also by {sorted(shared_developers)[0]}")

    return reasons


def leading_fields(score_parts):
    """Which of the three models actually drove this recommendation?

    We compare each model's *contribution* -- its score times its weight -- as a
    share of the total, rather than comparing the raw scores. A tag cosine and a
    description cosine are not comparable: they come from vocabularies of 449
    and 30,000 terms. Their contributions to one final score are.
    """
    contributions = {}
    for field, score in score_parts.items():
        contributions[field] = FIELD_WEIGHTS.get(field, 0.0) * score

    total = sum(contributions.values())
    if total <= 0:
        return []

    ranked = sorted(contributions, key=contributions.get, reverse=True)

    leading = []
    for field in ranked:
        if contributions[field] / total >= config.EXPLAIN_MIN_SHARE:
            leading.append(field)
    return leading


def shared_tags(query_game, recommended_game, tag_usage_counts):
    """Tags both games have, rarest first.

    Rarity is what makes a tag interesting -- the same idea as the IDF that
    weighted these tags in the first place. "Both are Indie" tells you nothing;
    "both are Deck Building Roguelikes" tells you everything.
    """
    query_tags = set(split_field(query_game, "tags"))
    recommended_tags = set(split_field(recommended_game, "tags"))
    shared = query_tags & recommended_tags

    # A tag we have no count for is brand new, so treat it as the rarest.
    return sorted(shared, key=lambda tag: tag_usage_counts.get(tag, 0))


def split_field(game, field):
    """Read a multi-value field, which may be a list or a comma-joined string."""
    value = game.get(field)
    if not value:
        return []
    if isinstance(value, str):
        return value.split(",")
    return list(value)
