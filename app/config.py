"""Every number you can tune lives here.

The reasoning behind these values is in docs/ENGINEERING.md.
"""

from pathlib import Path

# --- Where files live ----------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW_CSV = ROOT / "data" / "raw" / "games.csv"
SAMPLE_CSV = ROOT / "data" / "sample" / "games_sample.csv"
DATABASE = ROOT / "data" / "processed" / "catalogue.db"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"


# --- Which games make it into the catalogue ------------------------------

# Games with no reviews almost never have tags in this dataset (0.9% of them do,
# versus 100% of reviewed games), so we cannot recommend them at all.
MIN_REVIEWS = 10
MIN_DESCRIPTION_WORDS = 20
EXCLUDE_NAME_PATTERN = r"(?i)playtest|soundtrack|\bdemo\b|\bOST\b|artbook"


# --- Popularity score ----------------------------------------------------

WILSON_CONFIDENCE_Z = 1.96  # 95% confidence

# Popularity blends three things. The weights add up to 1.0.
QUALITY_WEIGHT = 0.60  # how well reviewed the game is
REACH_WEIGHT = 0.30  # how many people reviewed it
RECENCY_WEIGHT = 0.10  # how recently it came out

RECENCY_HALF_LIFE_YEARS = 3.0


# --- Similarity ----------------------------------------------------------

# We build three separate TF-IDF models and blend their similarity scores.
# These weights came out of the evaluation sweep, not out of thin air.
TAG_WEIGHT = 0.35
GENRE_WEIGHT = 0.20
DESCRIPTION_WEIGHT = 0.45

# How many similar games to pull out before reranking them.
CANDIDATE_COUNT = 300


# --- Reranking -----------------------------------------------------------

# final score = similarity * (floor + (1 - floor) * popularity)
# A floor of 0.70 means even a badly reviewed game keeps 70% of its similarity,
# so quality can reorder results but can never override relevance.
QUALITY_FLOOR = 0.70


# --- Explanations --------------------------------------------------------

# Only mention a field if it contributed at least this share of the score.
EXPLAIN_MIN_SHARE = 0.25

# What counts as "highly rated by the community".
ACCLAIM_MIN_REVIEWS = 500
ACCLAIM_MIN_RATIO = 0.90
