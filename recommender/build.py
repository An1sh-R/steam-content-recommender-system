"""Offline build: raw CSV -> processed catalogue.

python -m recommender.build --sample    # committed 600-game sample
python -m recommender.build             # full dataset
"""

from __future__ import annotations

import argparse

import pandas as pd

from recommender import catalogue, clean, config, documents, load, popularity, vectorize


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

    # The database is written first because it is the step that can fail on a
    # rebuild: a running API holds catalogue.db open, and Windows refuses to
    # replace it. Doing it before the vectors means that failure leaves the
    # artifacts untouched rather than newer than the database they index.
    catalogue.build_db(games, config.CATALOGUE_DB)

    matrices = vectorize.fit(documents.build_documents(games))
    for name, matrix in matrices.items():
        print(f"  {name:12s} {matrix.shape[0]:,} x {matrix.shape[1]:,}")
    vectorize.save(matrices, games["appid"].to_numpy(), config.ARTIFACTS_DIR)

    print(f"wrote {config.CATALOGUE_DB.name} and vectors in {config.ARTIFACTS_DIR.name}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="build from the committed sample")
    build(sample=parser.parse_args().sample)


if __name__ == "__main__":
    main()
