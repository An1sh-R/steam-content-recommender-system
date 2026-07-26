import pytest

from recommender import config, load


@pytest.fixture(scope="session")
def sample_df():
    return load.load_raw(config.SAMPLE_CSV)
