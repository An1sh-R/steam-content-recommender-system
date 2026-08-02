"""Turns the raw Steam CSV into everything the app needs to run.

    python -m app.build --sample    # the 600-game sample committed to this repo
    python -m app.build             # the full 125k-game dataset

It writes two things: the SQLite catalogue, and the TF-IDF matrices.
Nothing here runs while the app is serving requests.
"""

import argparse

import numpy as np
import pandas as pd

from app import config, database, recommender

# The published CSV header is broken: it lists 39 column names but every row has
# 40 values, because "Discount" and "DLC count" got glued together. So we ignore
# the header and name the columns ourselves, in the order they really appear.
# Get this wrong and "description" silently fills up with DLC counts.
RAW_COLUMNS = [
    "AppID", "Name", "Release date", "Estimated owners", "Peak CCU",
    "Required age", "Price", "Discount", "DLC count", "About the game",
    "Supported languages", "Full audio languages", "Reviews", "Header image",
    "Website", "Support url", "Support email", "Windows", "Mac", "Linux",
    "Metacritic score", "Metacritic url", "User score", "Positive", "Negative",
    "Score rank", "Achievements", "Recommendations", "Notes",
    "Average playtime forever", "Average playtime two weeks",
    "Median playtime forever", "Median playtime two weeks", "Developers",
    "Publishers", "Categories", "Genres", "Tags", "Screenshots", "Movies",
]

# The only columns we actually read, and what we call them afterwards.
COLUMNS_WE_KEEP = {
    "AppID": "appid",
    "Name": "name",
    "Release date": "release_date",
    "Estimated owners": "estimated_owners",
    "Price": "price",
    "About the game": "description",
    "Windows": "windows",
    "Mac": "mac",
    "Linux": "linux",
    "Positive": "positive",
    "Negative": "negative",
    "Developers": "developers",
    "Publishers": "publishers",
    "Categories": "categories",
    "Genres": "genres",
    "Tags": "tags",
}

# These arrive as "Action, Indie, RPG" and become ["Action", "Indie", "RPG"].
MULTI_VALUE_COLUMNS = ["categories", "genres", "tags"]


# --- Step 1: read the CSV ------------------------------------------------


def load_raw_csv(path):
    """Read the Steam dataset with the columns lined up correctly."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the Steam dataset there, or use the sample "
            f"that ships with this repo: {config.SAMPLE_CSV}"
        )

    games = pd.read_csv(
        path,
        header=0,  # skip the broken header line
        names=RAW_COLUMNS,  # and use our own names instead
        usecols=list(COLUMNS_WE_KEEP),
        index_col=False,
        low_memory=False,
    )
    games = games.rename(columns=COLUMNS_WE_KEEP)

    # Dates arrive as text, and in two different shapes: "Oct 21, 2008" and
    # "Oct 2008". Anything unparseable becomes NaT.
    games["release_date"] = pd.to_datetime(
        games["release_date"], errors="coerce", format="mixed"
    )

    # Missing text arrives as NaN. Without this the "has tags" check below would
    # let every untagged game through, because NaN != "" is True.
    text_columns = ["name", "description", "developers", "publishers"]
    for column in text_columns + MULTI_VALUE_COLUMNS:
        games[column] = games[column].fillna("").astype(str).str.strip()

    return games


# --- Step 2: clean it ----------------------------------------------------


def clean_games(raw_games):
    """Keep only the games we can actually recommend, and tidy up their fields."""
    games = raw_games.copy()
    review_count = games["positive"] + games["negative"]

    # A game needs enough reviews to judge, tags to match on, and a description
    # worth reading. We also drop soundtracks, demos and playtests, which are
    # not games you would want recommended.
    has_enough_reviews = review_count >= config.MIN_REVIEWS
    has_tags = games["tags"] != ""
    has_a_real_description = (
        games["description"].str.split().map(len) >= config.MIN_DESCRIPTION_WORDS
    )
    is_a_real_game = ~games["name"].str.contains(
        config.EXCLUDE_NAME_PATTERN, regex=True, na=False
    )

    games = games[
        has_enough_reviews & has_tags & has_a_real_description & is_a_real_game
    ].copy()

    for column in MULTI_VALUE_COLUMNS:
        games[column] = games[column].map(split_values)

    games["total_reviews"] = games["positive"] + games["negative"]
    games["release_year"] = games["release_date"].dt.year.astype("Int64")

    games = drop_duplicate_releases(games)
    return games.reset_index(drop=True)


def split_values(text):
    """"Action, Indie, Action" -> ["Action", "Indie"], keeping the original order."""
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part and part not in values:
            values.append(part)
    return values


def drop_duplicate_releases(games):
    """Collapse the same game listed under several AppIDs, keeping the main one.

    Steam carries old and regional re-releases: Portal 2 and Assassin's Creed 2
    each appear more than once. They are perfect matches for each other, so
    without this a game ends up recommending itself.

    We match on name AND description AND developer on purpose. Around 349 games
    share only a name with another game and are genuinely different games.
    """
    most_reviewed_first = games.sort_values("total_reviews", ascending=False)
    unique_games = most_reviewed_first.drop_duplicates(
        subset=["name", "description", "developers"]
    )
    return unique_games.sort_index()


# --- Step 3: score how popular each game is ------------------------------


def wilson_score(positive_reviews, negative_reviews, z=config.WILSON_CONFIDENCE_Z):
    """How good is this game, given how many people actually voted?

    Steam reviews are thumbs up or down, so "95% positive" means very different
    things with 20 reviews and with 20,000. This returns the lower end of a
    confidence interval instead, which pushes small samples down automatically:
    a single 5-star review scores 0.21, while 18,904 out of 19,064 scores 0.99.

    No minimum review count needed -- the maths handles it.
    """
    positive_reviews = np.asarray(positive_reviews, dtype=float)
    negative_reviews = np.asarray(negative_reviews, dtype=float)
    total = positive_reviews + negative_reviews

    # A game with no reviews scores 0 rather than dividing by zero.
    safe_total = np.where(total > 0, total, 1.0)
    observed_rate = positive_reviews / safe_total

    centre = observed_rate + z**2 / (2 * safe_total)
    spread = z * np.sqrt(
        observed_rate * (1 - observed_rate) / safe_total + z**2 / (4 * safe_total**2)
    )
    denominator = 1 + z**2 / safe_total

    lower_bound = np.clip((centre - spread) / denominator, 0.0, 1.0)
    return np.where(total > 0, lower_bound, 0.0)


def popularity_score(games, today=None):
    """Blend quality, reach and freshness into one 0-to-1 score.

    Wilson on its own flattens out above ~10,000 reviews, so a 99%-rated indie
    with 10k reviews would outrank a 96%-rated classic with a million. Adding
    reach and a little recency keeps the front page sensible.
    """
    if today is None:
        today = pd.Timestamp.now()

    quality = wilson_score(games["positive"], games["negative"])

    # Review counts run from 1 to 8.8 million, so we compare them on a log
    # scale -- otherwise only the top handful of games would score anything.
    review_counts = (games["positive"] + games["negative"]).to_numpy(dtype=float)
    reach = np.log1p(review_counts) / np.log1p(max(review_counts.max(), 1.0))

    age_in_years = (today - games["release_date"]).dt.days.to_numpy(dtype=float) / 365.25
    age_in_years = np.nan_to_num(age_in_years, nan=0.0).clip(min=0.0)
    recency = 0.5 ** (age_in_years / config.RECENCY_HALF_LIFE_YEARS)

    score = (
        config.QUALITY_WEIGHT * quality
        + config.REACH_WEIGHT * reach
        + config.RECENCY_WEIGHT * recency
    )
    return pd.Series(score, index=games.index, name="popularity")


# --- Putting it together -------------------------------------------------


def prepare_catalogue(raw_games):
    """Clean the raw data and attach a popularity score to every game."""
    games = clean_games(raw_games)
    games["popularity"] = popularity_score(games)
    return games


def main():
    parser = argparse.ArgumentParser(description="Build the game catalogue.")
    parser.add_argument(
        "--sample", action="store_true", help="build from the small sample dataset"
    )
    use_sample = parser.parse_args().sample

    source = config.SAMPLE_CSV if use_sample else config.RAW_CSV
    print(f"Reading {source.name} ...")
    raw_games = load_raw_csv(source)

    games = prepare_catalogue(raw_games)
    print(f"Keeping {len(games):,} of {len(raw_games):,} games.")

    # The database goes first because it is the step most likely to fail: if the
    # API is running it holds catalogue.db open and Windows will not let us
    # replace it. Failing here leaves the old matrices alone rather than leaving
    # them newer than the database they belong to.
    database.build_database(games, config.DATABASE)
    print(f"Wrote {config.DATABASE.name}")

    tag_matrix, genre_matrix, description_matrix = recommender.build_tfidf_matrices(games)
    for label, matrix in [
        ("tags", tag_matrix),
        ("genres", genre_matrix),
        ("description", description_matrix),
    ]:
        print(f"  {label:12s} {matrix.shape[0]:,} games x {matrix.shape[1]:,} terms")

    recommender.save_matrices(
        tag_matrix, genre_matrix, description_matrix,
        games["appid"].to_numpy(), config.ARTIFACTS_DIR,
    )
    print(f"Wrote matrices to {config.ARTIFACTS_DIR.name}/")


if __name__ == "__main__":
    main()
