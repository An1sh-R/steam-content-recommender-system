import numpy as np

from recommender import config, rerank


def test_quality_reorders_within_a_relevance_band():
    similarity = np.array([0.50, 0.49])
    scores = rerank.apply(similarity, np.array([0.0, 1.0]))
    assert scores[1] > scores[0]


def test_quality_cannot_promote_an_irrelevant_game():
    similarity = np.array([0.90, 0.40])
    scores = rerank.apply(similarity, np.array([0.0, 1.0]))
    assert scores[0] > scores[1]


def test_the_penalty_is_bounded_by_the_floor():
    similarity = np.array([1.0, 1.0])
    worst, best = rerank.apply(similarity, np.array([0.0, 1.0]))
    assert worst == config.QUALITY_FLOOR
    assert best == 1.0


def test_ranking_is_unchanged_when_quality_is_equal():
    similarity = np.array([0.9, 0.5, 0.1])
    scores = rerank.apply(similarity, np.full(3, 0.7))
    assert list(np.argsort(-scores)) == [0, 1, 2]
