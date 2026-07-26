import numpy as np
import pandas as pd
import pytest

from recommender import clean, popularity


def w(positive, negative):
    return float(popularity.wilson_lower_bound([positive], [negative])[0])


def test_no_reviews_scores_zero():
    assert w(0, 0) == 0.0


def test_small_samples_are_penalised():
    """The whole point: one 100% review is not evidence of quality."""
    assert w(1, 0) < 0.3
    assert w(10, 0) < w(100, 0) < w(10_000, 0)


def test_confidence_grows_towards_the_observed_rate():
    assert w(9_500, 500) == pytest.approx(0.95, abs=0.01)


def test_bounded_between_zero_and_one():
    scores = popularity.wilson_lower_bound([0, 1, 500, 10_000], [0, 99, 500, 10])
    assert ((scores >= 0) & (scores <= 1)).all()


def test_a_heavily_reviewed_good_game_beats_a_perfect_unknown():
    assert w(18_904, 160) > w(3, 0)


def test_more_negatives_lowers_the_score():
    assert w(100, 0) > w(100, 10) > w(100, 100)


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "positive": [10_000, 10_000, 50, 0],
            "negative": [100, 5_000, 5, 0],
            "release_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01", "2010-01-01"]
            ),
        }
    )


def test_popularity_is_bounded_and_ordered(frame):
    scores = popularity.popularity_score(frame, reference_date=pd.Timestamp("2025-01-01"))
    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores.iloc[0] > scores.iloc[1] > scores.iloc[3]


def test_recency_favours_newer_games_all_else_equal():
    df = pd.DataFrame(
        {
            "positive": [1_000, 1_000],
            "negative": [100, 100],
            "release_date": pd.to_datetime(["2024-01-01", "2005-01-01"]),
        }
    )
    scores = popularity.popularity_score(df, reference_date=pd.Timestamp("2025-01-01"))
    assert scores.iloc[0] > scores.iloc[1]


def test_unreleased_games_do_not_exceed_the_bound():
    df = pd.DataFrame(
        {
            "positive": [10],
            "negative": [0],
            "release_date": pd.to_datetime(["2030-01-01"]),
        }
    )
    scores = popularity.popularity_score(df, reference_date=pd.Timestamp("2025-01-01"))
    assert 0 <= scores.iloc[0] <= 1


def test_index_is_preserved(sample_df):
    games = clean.clean(sample_df)
    scores = popularity.popularity_score(games)
    assert scores.index.equals(games.index)
    assert not np.isnan(scores).any()
