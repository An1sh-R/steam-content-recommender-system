import numpy as np
import pandas as pd

from recommender import documents, vectorize


def test_one_document_per_space(games):
    docs = documents.build_documents(games)
    assert set(docs) == {"tags", "genres", "description"}
    assert all(len(series) == len(games) for series in docs.values())


def test_multiword_terms_stay_one_token():
    df = pd.DataFrame(
        {
            "tags": [["Turn-Based Strategy", "Open World"]],
            "genres": [["RPG"]],
            "categories": [["Single-player"]],
            "description": ["A game."],
        }
    )
    docs = documents.build_documents(df)

    assert docs["tags"].iloc[0] == "turn_based_strategy open_world"
    assert docs["genres"].iloc[0] == "rpg single_player"


def test_genres_space_includes_categories(games):
    docs = documents.build_documents(games)
    row = games.index[0]
    assert len(docs["genres"].loc[row].split()) == len(games["genres"].loc[row]) + len(
        games["categories"].loc[row]
    )


def test_publishers_never_enter_the_documents(games):
    """V1 indexed publishers by accident and became a publisher-matcher."""
    docs = documents.build_documents(games)
    publisher = games["publishers"].iloc[0].split(",")[0].lower().replace(" ", "_")

    assert publisher not in docs["tags"].iloc[0]
    assert publisher not in docs["genres"].iloc[0]


def test_vectorizing_gives_one_matrix_per_space(games):
    matrices = vectorize.fit(documents.build_documents(games))

    assert set(matrices) == {"tags", "genres", "description"}
    assert all(matrix.shape[0] == len(games) for matrix in matrices.values())


def test_rows_are_l2_normalised(games):
    """Retrieval relies on this: a dot product is then already the cosine."""
    matrices = vectorize.fit(documents.build_documents(games))
    tags = matrices["tags"]
    norms = np.asarray(tags.multiply(tags).sum(axis=1)).ravel()

    assert np.allclose(norms, 1.0)


def test_artifacts_round_trip(games, tmp_path):
    matrices = vectorize.fit(documents.build_documents(games))
    appids = games["appid"].to_numpy()

    vectorize.save(matrices, appids, tmp_path)
    loaded, loaded_appids = vectorize.load(tmp_path)

    assert (loaded_appids == appids).all()
    assert all((loaded[name] != matrices[name]).nnz == 0 for name in matrices)


def test_load_without_artifacts_is_a_clear_error(tmp_path):
    try:
        vectorize.load(tmp_path)
    except FileNotFoundError as error:
        assert "recommender.build" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError")
