"""Offline build: raw CSV -> processed catalogue.

python -m recommender.build --sample    # committed 600-game sample
python -m recommender.build             # full dataset
"""

from __future__ import annotations

import argparse

from recommender import catalogue, clean, config, load


def build(sample: bool = False) -> None:
    source = config.SAMPLE_CSV if sample else config.RAW_CSV
    print(f"loading {source.name} ...")
    raw = load.load_raw(source)

    games = clean.clean(raw)
    print(f"catalogue: {len(games):,} of {len(raw):,} games kept")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    games.to_parquet(config.GAMES_PARQUET, index=False)

    catalogue.build_db(games, config.CATALOGUE_DB)
    print(f"wrote {config.GAMES_PARQUET.name} and {config.CATALOGUE_DB.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="build from the committed sample")
    build(sample=parser.parse_args().sample)


if __name__ == "__main__":
    main()
