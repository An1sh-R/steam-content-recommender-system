"""The harness must be trustworthy before its numbers mean anything."""

import numpy as np
import pytest

from evaluation import metrics, protocol


def test_tag_split_is_disjoint_and_complete():
    tags = ["a", "b", "c", "d", "e"]
    feature, held = protocol.split_tags(tags)

    assert feature == ["a", "c", "e"]
    assert held == ["b", "d"]
    assert set(feature) | set(held) == set(tags)
    assert not set(feature) & set(held)


def test_held_out_tags_are_absent_from_the_features(games):
    """The protocol is worthless if the judging signal is in the feature space."""
    evaluated, _ = protocol.prepare(games)

    for original, feature in zip(games["tags"], evaluated["tags"], strict=True):
        _, held = protocol.split_tags(original)
        assert not set(feature) & set(held)


def test_prepare_does_not_mutate_the_input(games):
    before = games["tags"].map(list).tolist()
    protocol.prepare(games)
    assert games["tags"].tolist() == before


@pytest.fixture(scope="module")
def relevance(games):
    _, held_out = protocol.prepare(games)
    return protocol.Relevance(held_out)


def test_a_game_is_perfectly_relevant_to_itself(relevance):
    scores = relevance.against_all(0)
    assert scores[0] == pytest.approx(1.0)


def test_relevance_is_a_bounded_jaccard(relevance, games):
    scores = relevance.against_all(3)
    assert len(scores) == len(games)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_sampled_queries_are_valid_and_unique(games):
    queries = protocol.sample_queries(games, 40)
    assert len(set(queries.tolist())) == len(queries)
    assert all(len(games["tags"].iloc[q]) >= protocol.MIN_TAGS for q in queries)


def test_sampling_is_deterministic(games):
    assert (protocol.sample_queries(games, 30) == protocol.sample_queries(games, 30)).all()


# --- metrics -------------------------------------------------------------


def test_ndcg_is_one_for_a_perfect_ranking():
    truth = np.array([1.0, 0.8, 0.5, 0.2, 0.0])
    assert metrics.ndcg_at_k(np.array([1.0, 0.8, 0.5]), truth, k=3) == pytest.approx(1.0)


def test_ndcg_punishes_a_reversed_ranking():
    truth = np.array([1.0, 0.8, 0.5])
    perfect = metrics.ndcg_at_k(np.array([1.0, 0.8, 0.5]), truth, k=3)
    reversed_ = metrics.ndcg_at_k(np.array([0.5, 0.8, 1.0]), truth, k=3)
    assert reversed_ < perfect


def test_ndcg_is_zero_when_nothing_is_relevant():
    assert metrics.ndcg_at_k(np.zeros(5), np.zeros(5)) == 0.0


def test_recall_finds_the_ideal_top_ten():
    truth = np.zeros(100)
    truth[np.arange(10)] = 1.0  # rows 0-9 are the ideal set
    assert metrics.recall_at_k(np.arange(10), truth, k=50) == 1.0
    assert metrics.recall_at_k(np.arange(50, 60), truth, k=50) == 0.0


def test_tie_rate_detects_undiscriminating_scores():
    assert metrics.tie_rate(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.75)
    assert metrics.tie_rate(np.array([4.0, 3.0, 2.0, 1.0])) == 0.0


def test_novelty_is_higher_for_obscure_results():
    percentile = np.array([0.99, 0.98, 0.02, 0.01])
    blockbusters = metrics.novelty(np.array([0, 1]), percentile, k=2)
    obscure = metrics.novelty(np.array([2, 3]), percentile, k=2)
    assert obscure > blockbusters


def test_diversity_is_zero_for_identical_items(games):
    _, held_out = protocol.prepare(games)
    identical = held_out[[0, 0, 0]]
    assert metrics.intra_list_diversity(np.arange(3), identical, k=3) == pytest.approx(0.0)


def test_same_publisher_rate(games):
    publishers = np.array(["Valve", "Valve", "Ubisoft"])
    assert metrics.same_publisher_rate(np.array([1, 2]), publishers, 0, k=2) == 0.5
    assert metrics.same_publisher_rate(np.array([0]), np.array([""]), 0) is None
