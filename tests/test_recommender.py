"""The recommendation engine and its explanations."""

import numpy as np
import pandas as pd
import pytest

from app import config, explain, recommender

TAG_COUNTS = {"Indie": 40_000, "Action": 20_000, "Roguelike": 900, "Deck Building": 200}


# --- Turning games into text ---------------------------------------------


def test_each_game_gets_three_documents(games):
    tag_docs, genre_docs, description_docs = recommender.build_documents(games)
    assert len(tag_docs) == len(genre_docs) == len(description_docs) == len(games)

    # Genres and categories share a document, since both are short controlled lists.
    row = games.index[0]
    expected_terms = len(games["genres"].loc[row]) + len(games["categories"].loc[row])
    assert len(genre_docs.loc[row].split()) == expected_terms


def test_multi_word_tags_stay_a_single_term():
    """Otherwise "Turn-Based Strategy" matches anything that mentions strategy."""
    one_game = pd.DataFrame(
        {
            "tags": [["Turn-Based Strategy", "Open World"]],
            "genres": [["RPG"]],
            "categories": [["Single-player"]],
            "description": ["A game."],
        }
    )
    tag_docs, genre_docs, _ = recommender.build_documents(one_game)

    assert tag_docs.iloc[0] == "turn_based_strategy open_world"
    assert genre_docs.iloc[0] == "rpg single_player"


def test_publishers_never_reach_the_documents(games):
    """V1 indexed publishers by accident and turned into a publisher matcher."""
    tag_docs, genre_docs, _ = recommender.build_documents(games)
    publisher = games["publishers"].iloc[0].split(",")[0].lower().replace(" ", "_")

    assert publisher not in tag_docs.iloc[0]
    assert publisher not in genre_docs.iloc[0]


# --- The TF-IDF models ---------------------------------------------------


def test_every_matrix_has_a_row_per_game_and_unit_length_rows(games):
    """Unit-length rows are what let us use a dot product as the cosine."""
    matrices = recommender.build_tfidf_matrices(games)

    for matrix in matrices:
        assert matrix.shape[0] == len(games)
        lengths = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
        assert np.allclose(lengths, 1.0)


def test_matrices_survive_a_save_and_load(games, tmp_path):
    """The API loads these from disk, so a bad round trip breaks everything."""
    with pytest.raises(FileNotFoundError, match="app.build"):
        recommender.load_matrices(tmp_path)

    tag_matrix, genre_matrix, description_matrix = recommender.build_tfidf_matrices(games)
    appids = games["appid"].to_numpy()

    recommender.save_matrices(
        tag_matrix, genre_matrix, description_matrix, appids, tmp_path
    )
    loaded_tags, loaded_genres, loaded_descriptions, loaded_appids = (
        recommender.load_matrices(tmp_path)
    )

    assert (loaded_appids == appids).all()
    assert (loaded_tags != tag_matrix).nnz == 0
    assert (loaded_genres != genre_matrix).nnz == 0
    assert (loaded_descriptions != description_matrix).nnz == 0


# --- Similarity ----------------------------------------------------------


def test_a_game_is_perfectly_similar_to_itself(engine):
    scores = recommender.cosine_similarity_to_all(engine.tag_matrix, 0)
    assert scores[0] == pytest.approx(1.0)
    assert ((scores >= -1e-9) & (scores <= 1 + 1e-9)).all()


def test_scores_are_combined_using_the_weights_from_config():
    """Changing config must change the recommendations, or the config is a lie."""
    tags = np.array([1.0, 0.0, 0.5])
    genres = np.array([0.0, 1.0, 0.5])
    descriptions = np.array([0.0, 0.0, 1.0])

    given_weights = recommender.combine_scores(
        tags, genres, descriptions,
        tag_weight=0.5, genre_weight=0.2, description_weight=0.3,
    )
    # third game: 0.5*0.5 + 0.2*0.5 + 0.3*1.0
    assert np.allclose(given_weights, [0.5, 0.2, 0.65])

    # With no weights passed, the configured ones are used.
    configured = recommender.combine_scores(tags, genres, descriptions)
    assert configured[0] == pytest.approx(config.TAG_WEIGHT)
    assert configured[1] == pytest.approx(config.GENRE_WEIGHT)


def test_top_rows_returns_the_best_first_and_never_asks_for_too_many():
    scores = np.array([0.1, 0.9, 0.5, 0.7])

    assert list(recommender.top_rows(scores, 3)) == [1, 3, 2]
    # There are only three other games, however many we ask for.
    assert len(recommender.top_rows(scores, 99)) == 3


# --- Reranking -----------------------------------------------------------


def test_quality_reorders_similar_games_but_cannot_promote_an_irrelevant_one():
    """The floor is what stops popularity quietly taking over the ranking."""
    close_call = recommender.apply_quality_boost(
        np.array([0.50, 0.49]), np.array([0.0, 1.0])
    )
    assert close_call[1] > close_call[0], "quality breaks a near tie"

    not_close = recommender.apply_quality_boost(
        np.array([0.90, 0.40]), np.array([0.0, 1.0])
    )
    assert not_close[0] > not_close[1], "relevance still wins"

    # The worst possible game keeps exactly QUALITY_FLOOR of its similarity.
    worst, best = recommender.apply_quality_boost(
        np.array([1.0, 1.0]), np.array([0.0, 1.0])
    )
    assert worst == config.QUALITY_FLOOR
    assert best == 1.0


# --- End to end ----------------------------------------------------------


def test_recommend_returns_the_number_asked_for_with_no_duplicates(engine, games):
    """Also checks each result carries the numbers and reasons the UI needs."""
    appid = int(games["appid"].iloc[0])
    results = recommender.recommend(engine, appid, count=10)

    assert len(results) == 10
    assert len({game["appid"] for game in results}) == 10
    assert all(game["name"] and game["tags"] for game in results)

    for game in results:
        assert 0 <= game["similarity"] <= 1
        assert set(game["parts"]) == {"tags", "genres", "description"}
        assert game["reasons"]
        assert all(isinstance(reason, str) for reason in game["reasons"])


def test_a_game_is_never_its_own_recommendation(engine, games):
    """V1 dropped whatever came back first as 'the query' and leaked it into 22%
    of results. Checked across the catalogue, not just one game."""
    for position in range(0, len(games), 53):
        appid = int(games["appid"].iloc[position])
        results = recommender.recommend(engine, appid, count=10)
        assert all(game["appid"] != appid for game in results)


def test_recommendations_come_back_in_final_score_order(engine, games):
    """By final score, not raw similarity -- reranking is allowed to reorder them."""
    results = recommender.recommend(engine, int(games["appid"].iloc[0]), count=8)
    scores = [game["score"] for game in results]
    assert scores == sorted(scores, reverse=True)


def test_the_engine_knows_which_games_it_can_recommend(engine, games):
    assert engine.knows(int(games["appid"].iloc[0]))
    assert not engine.knows(-1)


# --- Explanations --------------------------------------------------------


@pytest.fixture
def query_game():
    return {
        "tags": "Indie,Action,Roguelike,Deck Building",
        "developers": "Mega Crit",
        "positive": 100,
        "total_reviews": 100,
    }


def a_candidate(**changes):
    game = {
        "tags": "Indie,Action,Roguelike,Deck Building",
        "developers": "Other Studio",
        "positive": 10,
        "total_reviews": 100,
    }
    game.update(changes)
    return game


def test_shared_tags_are_listed_rarest_first(query_game):
    """"Both are Indie" explains nothing. "Both are Deck Building" explains a lot."""
    shared = explain.shared_tags(query_game, a_candidate(), TAG_COUNTS)
    assert shared == ["Deck Building", "Roguelike", "Action", "Indie"]

    # A tag too new to have a count is treated as the rarest of all.
    brand_new = {"tags": "Indie,Roguelike,Brand New"}
    assert explain.shared_tags(brand_new, brand_new, TAG_COUNTS)[0] == "Brand New"


def test_a_model_is_only_named_when_it_actually_drove_the_score(query_game):
    """A description-driven match must not be explained as a tag match."""
    description_driven = {"tags": 0.01, "genres": 0.01, "description": 0.90}
    assert explain.explain_recommendation(
        query_game, a_candidate(), description_driven, TAG_COUNTS
    ) == ["Similar gameplay description"]

    tag_driven = {"tags": 0.90, "genres": 0.01, "description": 0.01}
    reasons = explain.explain_recommendation(
        query_game, a_candidate(), tag_driven, TAG_COUNTS
    )
    assert reasons[0] == "Shares 4 tags including Deck Building and Roguelike"

    # Two shared tags get named outright rather than counted.
    reasons = explain.explain_recommendation(
        query_game, a_candidate(tags="Roguelike,Deck Building"), tag_driven, TAG_COUNTS
    )
    assert reasons[0] == "Shares Deck Building and Roguelike"


def test_explanations_cope_with_nothing_matching_and_missing_metadata(query_game):
    no_similarity = {"tags": 0.0, "genres": 0.0, "description": 0.0}
    assert explain.explain_recommendation(
        query_game, a_candidate(), no_similarity, TAG_COUNTS
    ) == [], "say nothing rather than making something up"

    some_similarity = {"tags": 0.5, "genres": 0.5, "description": 0.5}
    bare_game = {"tags": None, "developers": None, "positive": 0, "total_reviews": 0}
    assert explain.explain_recommendation(
        query_game, bare_game, some_similarity, TAG_COUNTS
    )


def test_acclaim_needs_both_a_high_rating_and_enough_reviews(query_game):
    """Otherwise every 3-review game claims to be a community favourite."""
    parts = {"tags": 0.5, "genres": 0.5, "description": 0.5}
    acclaim = "Highly rated by the community"

    loved = a_candidate(positive=9_500, total_reviews=10_000)
    unproven = a_candidate(positive=10, total_reviews=10)
    mediocre = a_candidate(positive=5_000, total_reviews=10_000)

    assert acclaim in explain.explain_recommendation(query_game, loved, parts, TAG_COUNTS)
    assert acclaim not in explain.explain_recommendation(
        query_game, unproven, parts, TAG_COUNTS
    )
    assert acclaim not in explain.explain_recommendation(
        query_game, mediocre, parts, TAG_COUNTS
    )


def test_a_shared_studio_is_worth_mentioning_even_though_it_is_not_a_signal(query_game):
    """Developers are kept out of the models on purpose, but they are still a true
    and useful thing to say about a game that already earned its place."""
    parts = {"tags": 0.5, "genres": 0.0, "description": 0.0}
    reasons = explain.explain_recommendation(
        query_game, a_candidate(developers="Mega Crit"), parts, TAG_COUNTS
    )
    assert "Also by Mega Crit" in reasons


