"""Every tunable in the project lives here."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "games.csv"
SAMPLE_CSV = DATA_DIR / "sample" / "games_sample.csv"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

CATALOGUE_DB = PROCESSED_DIR / "catalogue.db"

# --- Catalogue filter (see clean.py) -------------------------------------
# Games below this review count have no tags in the source data (0.9% coverage
# vs 100% above it), so they are unrecommendable rather than merely unpopular.
MIN_REVIEWS = 10
MIN_DESCRIPTION_WORDS = 20
EXCLUDE_NAME_PATTERN = r"(?i)playtest|soundtrack|\bdemo\b|\bOST\b|artbook"

# --- Popularity score (see popularity.py) --------------------------------
WILSON_Z = 1.96  # 95% confidence
POPULARITY_WEIGHTS = {"quality": 0.60, "reach": 0.30, "recency": 0.10}
RECENCY_HALF_LIFE_YEARS = 3.0

# --- Similarity (see documents.py / retrieval.py) ------------------------
# One TF-IDF space per field group, combined by weighted cosine.
#
# Set by the M4 evaluation harness, not by hand. My prior was 0.60/0.15/0.25 on
# the reasoning that curated tags beat marketing prose; the sweep disagreed and
# descriptions earn nearly as much weight as tags (NDCG@10 0.247 -> 0.256,
# significant on a held-out query sample). The surface is flat, though: every
# sensible split lands within ~5%, while dropping to a single concatenated
# space costs 46%. Having three spaces matters far more than their exact ratio.
FIELD_WEIGHTS = {"tags": 0.35, "genres": 0.20, "description": 0.45}
N_CANDIDATES = 300
