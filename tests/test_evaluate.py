"""The evaluation harness.

The numbers this produces are the main evidence that the recommender works, so
the harness itself has to be trustworthy first.
"""

import numpy as np
import pytest

from app import evaluate


@pytest.fixture(scope="module")
def data(games):
    return evaluate.prepare_evaluation_data(games)


# --- The held-out tag protocol -------------------------------------------


def test_the_two_halves_of_a_tag_list_do_not_overlap():
    tags = ["a", "b", "c", "d", "e"]
    training, held_out = evaluate.split_tags(tags)

    assert training == ["a", "c", "e"]
    assert held_out == ["b", "d"]
    assert set(training) | set(held_out) == set(tags)
    assert not set(training) & set(held_out)


def test_the_answers_are_never_in_the_training_data(games, data):
    """If the models could see the tags they are judged on, the whole evaluation
    would be meaningless. This is the test the protocol lives or dies by."""
    for original, training in zip(games["tags"], data.games_for_training["tags"]):
        _, held_out = evaluate.split_tags(original)
        assert not set(training) & set(held_out)

    # And hiding the tags must not damage the real catalogue.
    before = games["tags"].map(list).tolist()
    evaluate.prepare_evaluation_data(games)
    assert games["tags"].tolist() == before


def test_relevance_is_a_jaccard_score_against_the_hidden_tags(data, games):
    scores = evaluate.true_relevance(data, 0)

    assert len(scores) == len(games)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores[0] == pytest.approx(1.0), "a game shares every hidden tag with itself"


def test_query_games_are_valid_and_the_same_every_run(games):
    """Stratified across popularity, so the numbers describe games people search
    for rather than only the long tail."""
    queries = evaluate.choose_query_games(games, 40)

    assert len(set(queries.tolist())) == len(queries)
    assert all(
        len(games["tags"].iloc[q]) >= evaluate.MIN_TAGS_TO_BE_A_QUERY for q in queries
    )
    assert (queries == evaluate.choose_query_games(games, 40)).all()


# --- The metrics ---------------------------------------------------------


def test_precision_recall_and_map_agree_on_a_known_ranking():
    relevant = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    perfect = np.arange(10)
    assert evaluate.precision_at_k(perfect, relevant) == 1.0
    assert evaluate.recall_at_k(perfect, relevant, k=50) == 1.0
    assert evaluate.average_precision(perfect, relevant) == pytest.approx(1.0)

    nothing_relevant = np.arange(50, 60)
    assert evaluate.precision_at_k(nothing_relevant, relevant) == 0.0
    assert evaluate.recall_at_k(nothing_relevant, relevant, k=50) == 0.0
    assert evaluate.average_precision(nothing_relevant, relevant) == 0.0

    # MAP is the one that cares where in the list the good results landed.
    early = evaluate.average_precision(np.array([0, 1, 8, 9]), {0, 1})
    late = evaluate.average_precision(np.array([8, 9, 0, 1]), {0, 1})
    assert early > late


def test_ndcg_is_one_for_a_perfect_ranking_and_lower_for_a_worse_one():
    relevance = np.array([1.0, 0.8, 0.5, 0.2, 0.0])

    perfect = evaluate.ndcg_at_k(np.array([0, 1, 2]), relevance, k=3)
    backwards = evaluate.ndcg_at_k(np.array([2, 1, 0]), relevance, k=3)

    assert perfect == pytest.approx(1.0)
    assert backwards < perfect
    assert evaluate.ndcg_at_k(np.arange(5), np.zeros(5)) == 0.0


def test_the_beyond_accuracy_metrics_measure_what_they_claim_to():
    # A model that cannot tell games apart scores badly on ties.
    assert evaluate.tie_rate(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.75)
    assert evaluate.tie_rate(np.array([4.0, 3.0, 2.0, 1.0])) == 0.0

    # Novelty is higher when the results are not all blockbusters.
    percentile = np.array([0.99, 0.98, 0.02, 0.01])
    blockbusters = evaluate.novelty(np.array([0, 1]), percentile, k=2)
    obscure = evaluate.novelty(np.array([2, 3]), percentile, k=2)
    assert obscure > blockbusters

    # Two of these four games are rated below 70%.
    ratios = np.array([0.95, 0.40, 0.30, 0.99])
    assert evaluate.poorly_rated_rate(np.arange(4), ratios, k=4) == 0.5


def test_diversity_is_zero_when_every_result_is_the_same_game(data):
    identical = data.held_out_tags[[0, 0, 0]]
    assert evaluate.intra_list_diversity(np.arange(3), identical, k=3) == pytest.approx(0.0)


def test_same_publisher_rate_catches_a_publisher_matcher(data):
    publishers = np.array(["Valve", "Valve", "Ubisoft"])
    assert evaluate.same_publisher_rate(np.array([1, 2]), publishers, 0, k=2) == 0.5
    # Nothing to say when the query has no publisher listed.
    assert evaluate.same_publisher_rate(np.array([0]), np.array([""]), 0) is None
