import numpy as np
import pytest

from recommender import catalogue, documents, retrieval, vectorize
from recommender.engine import Engine


@pytest.fixture(scope="module")
def matrices(games):
    return vectorize.fit(documents.build_documents(games))


@pytest.fixture(scope="module")
def engine(games, matrices, tmp_path_factory):
    path = tmp_path_factory.mktemp("engine") / "catalogue.db"
    catalogue.build_db(games, path)
    return Engine(matrices, games["appid"].to_numpy(), catalogue.connect(path))


def test_query_is_never_its_own_recommendation(matrices, games):
    """V1 dropped rank 0 as 'self' and leaked the query into 22% of results."""
    for row in range(0, len(games), 37):
        top, _, _ = retrieval.similar_rows(matrices, row, n=10)
        assert row not in top


def test_results_are_sorted_by_descending_score(matrices):
    _, scores, _ = retrieval.similar_rows(matrices, 0, n=25)
    assert list(scores) == sorted(scores, reverse=True)


def test_returns_requested_count(matrices):
    top, scores, parts = retrieval.similar_rows(matrices, 0, n=15)
    assert len(top) == len(scores) == 15
    assert all(len(values) == 15 for values in parts.values())


def test_breakdown_has_one_score_per_space(matrices):
    _, _, parts = retrieval.similar_rows(matrices, 0, n=5)
    assert set(parts) == {"tags", "genres", "description"}
    assert all(((values >= -1e-9) & (values <= 1 + 1e-9)).all() for values in parts.values())


def test_combined_score_is_the_weighted_sum_of_parts(matrices):
    weights = {"tags": 0.6, "genres": 0.15, "description": 0.25}
    _, scores, parts = retrieval.similar_rows(matrices, 0, n=5, weights=weights)

    expected = sum(weights[name] * values for name, values in parts.items())
    assert np.allclose(scores, expected)


def test_weights_change_the_ranking(matrices):
    tags_only, _, _ = retrieval.similar_rows(matrices, 0, n=10, weights={"tags": 1.0})
    text_only, _, _ = retrieval.similar_rows(matrices, 0, n=10, weights={"description": 1.0})
    assert list(tags_only) != list(text_only)


def test_n_larger_than_catalogue_is_clamped(matrices, games):
    top, _, _ = retrieval.similar_rows(matrices, 0, n=len(games) + 500)
    assert len(top) == len(games)


def test_engine_returns_hydrated_games(engine, games):
    appid = int(games["appid"].iloc[0])
    results = engine.similar(appid, k=5)

    assert len(results) == 5
    assert all(result["name"] and result["tags"] for result in results)
    assert all(appid != result["appid"] for result in results)


def test_engine_attaches_similarity_and_breakdown(engine, games):
    result = engine.similar(int(games["appid"].iloc[0]), k=3)[0]

    assert 0 <= result["similarity"] <= 1
    assert set(result["parts"]) == {"tags", "genres", "description"}


def test_engine_results_stay_in_rank_order(engine, games):
    """By final score, not raw similarity -- the quality prior reorders them."""
    results = engine.similar(int(games["appid"].iloc[0]), k=8)
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_engine_explains_every_recommendation(engine, games):
    for result in engine.similar(int(games["appid"].iloc[0]), k=10):
        assert result["reasons"]
        assert all(isinstance(reason, str) for reason in result["reasons"])


def test_engine_knows_which_games_it_has(engine, games):
    assert engine.knows(int(games["appid"].iloc[0]))
    assert not engine.knows(-1)
