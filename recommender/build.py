"""Offline build: raw CSV -> processed catalogue.

python -m recommender.build --sample    # committed 600-game sample
python -m recommender.build             # full dataset
"""

from __future__ import annotations

import argparse

import pandas as pd

from recommender import catalogue, clean, config, load, popularity


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaned catalogue with derived scores attached, ready for indexing."""
    games = clean.clean(raw)
    games["popularity"] = popularity.popularity_score(games)
    return games


def build(sample: bool = False) -> None:
    source = config.SAMPLE_CSV if sample else config.RAW_CSV
    print(f"loading {source.name} ...")
    raw = load.load_raw(source)

    # The cleaned frame stays in memory and is handed to each build step in
    # turn; nothing at serve time reads it, so it is never serialised.
    games = prepare(raw)
    print(f"catalogue: {len(games):,} of {len(raw):,} games kept")

    catalogue.build_db(games, config.CATALOGUE_DB)
    print(f"wrote {config.CATALOGUE_DB.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="build from the committed sample")
    build(sample=parser.parse_args().sample)


if __name__ == "__main__":
    main()
