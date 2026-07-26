"""Semantic checks that each column holds what its name claims.

These are the real guard against the header defect: a misaligned read still
produces a valid-looking DataFrame, so shape assertions catch nothing. Only
asserting on *content* does.
"""

import pandas as pd

from recommender import config, schema

KNOWN_TAGS = {"Singleplayer", "Indie", "Atmospheric", "2D", "Action"}
KNOWN_GENRES = {"Action", "Adventure", "RPG", "Indie", "Casual", "Strategy"}
KNOWN_CATEGORIES = {"Single-player", "Steam Achievements", "Family Sharing"}


def _values(series: pd.Series) -> set[str]:
    out: set[str] = set()
    for cell in series.dropna():
        out.update(part.strip() for part in str(cell).split(",") if part.strip())
    return out


def test_expected_columns_present(sample_df):
    assert set(sample_df.columns) == set(schema.COLUMN_RENAME.values())


def test_appids_are_unique_integers(sample_df):
    assert sample_df["appid"].notna().all()
    assert not sample_df["appid"].duplicated().any()


def test_description_is_prose_not_a_number(sample_df):
    """Under the defect this column holds DLC count, an integer."""
    described = sample_df.loc[sample_df["description"] != "", "description"]
    assert len(described) > 100
    assert described.str.split().str.len().mean() > 50


def test_tags_contain_real_steam_tags(sample_df):
    assert KNOWN_TAGS & _values(sample_df["tags"])


def test_genres_contain_real_genres(sample_df):
    assert KNOWN_GENRES & _values(sample_df["genres"])


def test_categories_hold_platform_features_not_publishers(sample_df):
    """Under the defect this column holds Publishers."""
    assert KNOWN_CATEGORIES & _values(sample_df["categories"])


def test_review_counts_are_counts_not_percentages(sample_df):
    """Under the defect `positive` holds User score, which caps at 100."""
    assert sample_df["positive"].max() > 1000


def test_header_image_is_a_url(sample_df):
    urls = sample_df.loc[sample_df["header_image"] != "", "header_image"]
    assert (urls.str.startswith("http")).mean() > 0.95


def test_types_are_coerced(sample_df):
    assert pd.api.types.is_bool_dtype(sample_df["windows"])
    assert pd.api.types.is_numeric_dtype(sample_df["price"])
    assert pd.api.types.is_datetime64_any_dtype(sample_df["release_date"])
    assert sample_df["windows"].any()


def test_naive_read_is_misaligned(sample_df):
    """Demonstrates the defect: trusting the published header mislabels columns."""
    naive = pd.read_csv(config.SAMPLE_CSV, index_col=False, low_memory=False)

    naive_desc = naive["About the game"].dropna().astype(str)
    assert naive_desc.str.split().str.len().mean() < 5, "naive read should NOT yield prose"

    # Ours does, from the same file.
    assert sample_df["description"].str.split().str.len().mean() > 50
