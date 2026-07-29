"""An explanation that is not backed by the score is worse than no explanation."""

import pytest

from recommender import config, explain

TAG_COUNTS = {"Indie": 40_000, "Action": 20_000, "Roguelike": 900, "Deck Building": 200}


@pytest.fixture
def query():
    return {
        "tags": "Indie,Action,Roguelike,Deck Building",
        "developers": "Mega Crit",
        "positive": 100,
        "total_reviews": 100,
    }


def _candidate(**overrides):
    game = {
        "tags": "Indie,Action,Roguelike,Deck Building",
        "developers": "Other Studio",
        "positive": 10,
        "total_reviews": 100,
    }
    return {**game, **overrides}


def test_shared_tags_are_ordered_by_rarity(query):
    """The rarest shared tag is the informative one. 'Both are Indie' says nothing."""
    shared = explain.shared_tags(query, _candidate(), TAG_COUNTS)
    assert shared == ["Deck Building", "Roguelike", "Action", "Indie"]


def test_a_tag_missing_from_the_counts_sorts_as_rarest():
    """A tag too new to have a count is treated as maximally informative."""
    both = {"tags": "Indie,Roguelike,Brand New"}
    assert explain.shared_tags(both, both, TAG_COUNTS)[0] == "Brand New"


def test_a_field_is_only_named_when_it_drove_the_score(query):
    """Descriptions carry the most weight, so a description-driven match must not
    be explained as a tag match."""
    parts = {"tags": 0.01, "genres": 0.01, "description": 0.90}
    lines = explain.reasons(query, _candidate(), parts, TAG_COUNTS)
    assert lines == ["Similar gameplay description"]


def test_a_tag_driven_match_names_the_tags(query):
    parts = {"tags": 0.90, "genres": 0.01, "description": 0.01}
    lines = explain.reasons(query, _candidate(), parts, TAG_COUNTS)
    assert lines[0] == "Shares 4 tags including Deck Building and Roguelike"


def test_two_shared_tags_are_named_without_a_count(query):
    parts = {"tags": 0.90, "genres": 0.0, "description": 0.0}
    candidate = _candidate(tags="Roguelike,Deck Building")
    lines = explain.reasons(query, candidate, parts, TAG_COUNTS)
    assert lines[0] == "Shares Deck Building and Roguelike"


def test_no_reasons_when_nothing_matched(query):
    parts = dict.fromkeys(config.FIELD_WEIGHTS, 0.0)
    assert explain.reasons(query, _candidate(), parts, TAG_COUNTS) == []


def test_acclaim_needs_both_a_high_ratio_and_enough_reviews(query):
    parts = {"tags": 0.5, "genres": 0.5, "description": 0.5}
    loved = _candidate(positive=9_500, total_reviews=10_000)
    unproven = _candidate(positive=10, total_reviews=10)
    mediocre = _candidate(positive=5_000, total_reviews=10_000)

    assert "Highly rated by the community" in explain.reasons(query, loved, parts, TAG_COUNTS)
    assert "Highly rated by the community" not in explain.reasons(
        query, unproven, parts, TAG_COUNTS
    )
    assert "Highly rated by the community" not in explain.reasons(
        query, mediocre, parts, TAG_COUNTS
    )


def test_a_shared_developer_is_an_explanation_not_a_ranking_signal(query):
    """Developers are excluded from similarity (see DEVELOPMENT.md 6.5) but they are a
    true and useful thing to say about a game that already ranked."""
    parts = {"tags": 0.5, "genres": 0.0, "description": 0.0}
    lines = explain.reasons(query, _candidate(developers="Mega Crit"), parts, TAG_COUNTS)
    assert "Also by Mega Crit" in lines


def test_reasons_survive_missing_metadata(query):
    parts = {"tags": 0.5, "genres": 0.5, "description": 0.5}
    bare = {"tags": None, "developers": None, "positive": 0, "total_reviews": 0}
    assert explain.reasons(query, bare, parts, TAG_COUNTS)
