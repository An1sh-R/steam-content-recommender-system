import pandas as pd
import pytest

from recommender import clean, config


@pytest.fixture(scope="session")
def clean_df(sample_df):
    return clean.clean(sample_df)


def test_filter_removes_games(clean_df, sample_df):
    assert 0 < len(clean_df) < len(sample_df)


def test_every_kept_game_meets_the_thresholds(clean_df):
    assert (clean_df["total_reviews"] >= config.MIN_REVIEWS).all()
    assert clean_df["tags"].map(len).gt(0).all()
    assert clean_df["description"].str.split().str.len().ge(config.MIN_DESCRIPTION_WORDS).all()


def test_junk_titles_are_excluded(clean_df):
    assert not clean_df["name"].str.contains(config.EXCLUDE_NAME_PATTERN, regex=True).any()


def test_multivalue_fields_become_lists(clean_df):
    row = clean_df.iloc[0]
    assert isinstance(row["tags"], list)
    assert isinstance(row["genres"], list)
    assert isinstance(row["categories"], list)
    assert all(isinstance(tag, str) and tag == tag.strip() for tag in row["tags"])


def test_derived_columns(clean_df):
    assert (clean_df["total_reviews"] == clean_df["positive"] + clean_df["negative"]).all()
    assert clean_df["release_year"].between(1990, 2030).all()


def test_index_is_reset(clean_df):
    assert clean_df.index.tolist() == list(range(len(clean_df)))


def test_appids_still_unique(clean_df):
    assert not clean_df["appid"].duplicated().any()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Action, Indie ,RPG", ["Action", "Indie", "RPG"]),
        ("Action,Action,Indie", ["Action", "Indie"]),  # de-duplicated, order kept
        ("", []),
        (" , ,", []),
    ],
)
def test_split_values(raw, expected):
    assert clean.split_values(raw) == expected


def test_clean_does_not_mutate_input(sample_df):
    before = sample_df.copy()
    clean.clean(sample_df)
    pd.testing.assert_frame_equal(sample_df, before)
