import pytest

from recommender import build, config, load


@pytest.fixture(scope="session")
def sample_df():
    return load.load_raw(config.SAMPLE_CSV)


@pytest.fixture(scope="session")
def games(sample_df):
    """The catalogue as the real build produces it: cleaned + scored."""
    return build.prepare(sample_df)
