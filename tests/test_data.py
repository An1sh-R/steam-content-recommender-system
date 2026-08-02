"""Reading and cleaning the Steam CSV.

The published header is broken, and getting it wrong is silent: the app still
runs, it just recommends nonsense because "description" is full of DLC counts.
Most of these tests exist to make sure that never happens again.
"""

import csv

import pandas as pd
import pytest

from app import build, config

REAL_TAGS = {"Singleplayer", "Indie", "Atmospheric", "2D", "Action"}
REAL_CATEGORIES = {"Single-player", "Steam Achievements", "Family Sharing"}


def all_values(column):
    """Every distinct comma-separated value in a column."""
    values = set()
    for cell in column.dropna():
        for part in str(cell).split(","):
            if part.strip():
                values.add(part.strip())
    return values


# --- Reading the CSV -----------------------------------------------------


def test_the_published_header_really_is_broken():
    """39 column names, 40 values per row. This is why we name columns ourselves."""
    with open(config.SAMPLE_CSV, encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        row_widths = {len(row) for row in reader}

    assert len(header) == 39
    assert row_widths == {40}
    assert header[7] == "DiscountDLC count", "the two glued columns"

    # Our layout splits exactly that field back into two.
    assert len(build.RAW_COLUMNS) == 40
    assert build.RAW_COLUMNS[7] == "Discount"
    assert build.RAW_COLUMNS[8] == "DLC count"


def test_columns_hold_what_their_names_say(raw_games):
    """The whole point of RAW_COLUMNS. Each of these fails under the naive read."""
    assert set(raw_games.columns) == set(build.COLUMNS_WE_KEEP.values())

    # description is prose, not a DLC count
    described = raw_games.loc[raw_games["description"] != "", "description"]
    assert described.str.split().str.len().mean() > 50

    # categories holds platform features, not publishers
    assert REAL_CATEGORIES & all_values(raw_games["categories"])

    # tags holds real Steam tags
    assert REAL_TAGS & all_values(raw_games["tags"])

    # positive is a review count, not a 0-100 score
    assert raw_games["positive"].max() > 1000

    # and the types are usable downstream
    assert not raw_games["appid"].duplicated().any()
    assert pd.api.types.is_numeric_dtype(raw_games["price"])
    assert pd.api.types.is_datetime64_any_dtype(raw_games["release_date"])
    assert pd.api.types.is_bool_dtype(raw_games["windows"])


def test_trusting_the_published_header_gives_garbage():
    """Demonstrates the bug this module exists to avoid."""
    naive = pd.read_csv(config.SAMPLE_CSV, index_col=False, low_memory=False)
    naive_descriptions = naive["About the game"].dropna().astype(str)

    assert naive_descriptions.str.split().str.len().mean() < 5


def test_missing_csv_says_what_to_do():
    with pytest.raises(FileNotFoundError, match="sample"):
        build.load_raw_csv(config.ROOT / "data" / "nope.csv")


# --- Cleaning ------------------------------------------------------------


def test_only_recommendable_games_survive(games, raw_games):
    """Every filter in clean_games, checked on its output."""
    assert 0 < len(games) < len(raw_games)
    assert (games["total_reviews"] >= config.MIN_REVIEWS).all()
    assert games["tags"].map(len).gt(0).all()
    assert (
        games["description"].str.split().str.len() >= config.MIN_DESCRIPTION_WORDS
    ).all()
    assert not games["name"].str.contains(config.EXCLUDE_NAME_PATTERN, regex=True).any()


def test_cleaning_derives_the_columns_the_app_needs(games):
    assert (games["total_reviews"] == games["positive"] + games["negative"]).all()
    assert games["release_year"].between(1990, 2030).all()
    assert games.index.tolist() == list(range(len(games)))
    assert not games["appid"].duplicated().any()

    first = games.iloc[0]
    assert isinstance(first["tags"], list)
    assert isinstance(first["genres"], list)
    assert all(tag == tag.strip() for tag in first["tags"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Action, Indie ,RPG", ["Action", "Indie", "RPG"]),
        ("Action,Action,Indie", ["Action", "Indie"]),  # duplicates dropped, order kept
        ("", []),
        (" , ,", []),
    ],
)
def test_split_values(raw, expected):
    assert build.split_values(raw) == expected


def test_re_releases_are_collapsed_keeping_the_most_reviewed():
    """Steam lists Portal 2 under several AppIDs. Without this it recommends itself."""
    listings = pd.DataFrame(
        {
            "appid": [1, 2, 3],
            "name": ["Portal 2", "Portal 2", "Portal 2"],
            "description": ["a puzzle game", "a puzzle game", "a different game"],
            "developers": ["Valve", "Valve", "Valve"],
            "total_reviews": [100, 900, 50],
        }
    )
    kept = build.drop_duplicate_releases(listings)

    # The duplicate goes, the busiest listing wins, the real other game stays.
    assert kept["appid"].tolist() == [2, 3]


def test_cleaning_does_not_change_the_input(raw_games):
    before = raw_games.copy()
    build.clean_games(raw_games)
    pd.testing.assert_frame_equal(raw_games, before)


# --- Scoring -------------------------------------------------------------


def wilson(positive, negative):
    return float(build.wilson_score([positive], [negative])[0])


def test_wilson_punishes_small_samples_and_converges_on_the_real_rate():
    """The whole reason we use Wilson: one 100% review is not evidence of quality."""
    assert wilson(0, 0) == 0.0
    assert wilson(1, 0) < 0.3, "a single perfect review proves little"
    assert wilson(10, 0) < wilson(100, 0) < wilson(10_000, 0), "confidence grows"
    assert wilson(18_904, 160) > wilson(3, 0), "a proven hit beats a perfect unknown"

    # With plenty of reviews it settles on roughly the observed rate.
    assert wilson(9_500, 500) == pytest.approx(0.95, abs=0.01)
    assert wilson(100, 0) > wilson(100, 10) > wilson(100, 100)

    scores = build.wilson_score([0, 1, 500, 10_000], [0, 99, 500, 10])
    assert ((scores >= 0) & (scores <= 1)).all()


def test_popularity_blends_quality_reach_and_recency():
    listings = pd.DataFrame(
        {
            "positive": [10_000, 10_000, 1_000, 1_000],
            "negative": [100, 5_000, 100, 100],
            "release_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01", "2005-01-01"]
            ),
        }
    )
    scores = build.popularity_score(listings, today=pd.Timestamp("2025-01-01"))

    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores.iloc[0] > scores.iloc[1], "better reviewed wins"
    assert scores.iloc[0] > scores.iloc[2], "more reviewed wins"
    assert scores.iloc[2] > scores.iloc[3], "newer wins, all else equal"


def test_popularity_survives_odd_dates_and_lines_up_with_the_catalogue(games):
    """A release date in the future must not give a negative age or a score > 1."""
    unreleased = pd.DataFrame(
        {
            "positive": [10],
            "negative": [0],
            "release_date": pd.to_datetime(["2030-01-01"]),
        }
    )
    score = build.popularity_score(unreleased, today=pd.Timestamp("2025-01-01"))
    assert 0 <= score.iloc[0] <= 1

    # And on real data every game gets a usable score.
    assert games["popularity"].notna().all()
    assert games["popularity"].between(0, 1).all()
