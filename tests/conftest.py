"""Shared test fixtures.

Everything is built from the 600-game sample that ships with the repo, so the
whole suite runs in a few seconds and needs no downloads.
"""

import pytest

from app import build, config, database, recommender


@pytest.fixture(scope="session")
def raw_games():
    """The sample CSV, straight off disk."""
    return build.load_raw_csv(config.SAMPLE_CSV)


@pytest.fixture(scope="session")
def games(raw_games):
    """The catalogue exactly as the real build produces it: cleaned and scored."""
    return build.prepare_catalogue(raw_games)


@pytest.fixture(scope="session")
def connection(games, tmp_path_factory):
    """A real SQLite catalogue built from the sample."""
    path = tmp_path_factory.mktemp("catalogue") / "catalogue.db"
    database.build_database(games, path)
    return database.connect(path)


@pytest.fixture(scope="session")
def engine(games, connection):
    """A working Engine, with the TF-IDF models fitted on the sample."""
    tag_matrix, genre_matrix, description_matrix = recommender.build_tfidf_matrices(games)
    return recommender.Engine(
        tag_matrix,
        genre_matrix,
        description_matrix,
        games["appid"].to_numpy(),
        connection,
    )
