"""The systems being compared.

A model is just a name and a ranking function, so baselines and the real thing
go through identical measurement code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from recommender import config, documents, rerank, retrieval, vectorize


@dataclass
class Model:
    name: str
    note: str
    rank: Callable[[int, int], tuple[np.ndarray, np.ndarray]]


def build(
    games: pd.DataFrame,
    weight_sweep: dict[str, dict[str, float]],
    stage_sweep: dict[str, tuple[str, dict]] | None = None,
) -> list[Model]:
    """Baselines, the field-weighted model and its ablations, and the M5 stages."""
    docs = documents.build_documents(games)
    matrices = vectorize.fit(docs)
    popularity = games["popularity"].to_numpy()
    n_games = len(games)

    # The design decision under test: three weighted spaces, or one big document?
    merged = TfidfVectorizer(stop_words="english", max_features=30_000).fit_transform(
        docs["tags"] + " " + docs["genres"] + " " + docs["description"]
    )

    models = [
        Model("random", "sanity floor", _random_ranker(n_games)),
        Model("popularity", "same list for every query", _popularity_ranker(popularity)),
        Model("single space", "everything concatenated", _tfidf_ranker({"all": merged})),
    ]
    models += [
        Model(f"{space} only", "single field", _tfidf_ranker(matrices, {space: 1.0}))
        for space in ("genres", "description", "tags")
    ]
    models += [
        Model(f"weighted {name}", _describe(weights), _tfidf_ranker(matrices, weights))
        for name, weights in weight_sweep.items()
    ]
    models += [
        Model(name, note, _pipeline_ranker(matrices, popularity, **stage))
        for name, (note, stage) in (stage_sweep or {}).items()
    ]
    return models


def _describe(weights: dict[str, float]) -> str:
    return " / ".join(f"{name[0]}{value:.2f}" for name, value in weights.items())


def _tfidf_ranker(matrices, weights=None):
    weights = weights or dict.fromkeys(matrices, 1.0)

    def rank(row: int, n: int):
        rows, scores, _ = retrieval.similar_rows(matrices, row, n=n, weights=weights)
        return rows, scores

    return rank


def _pipeline_ranker(matrices, popularity: np.ndarray, quality: bool):
    """The shipped pipeline, with the quality prior switchable so it can be ablated.

    Retrieval always widens to N_CANDIDATES first; the stage under test only
    reorders that pool, which is exactly what production does.
    """

    def rank(row: int, n: int):
        rows, scores, _ = retrieval.similar_rows(matrices, row, n=config.N_CANDIDATES)
        if quality:
            scores = rerank.apply(scores, popularity[rows])
        picked = np.argsort(-scores)[:n]
        return rows[picked], scores[picked]

    return rank


def _popularity_ranker(popularity: np.ndarray):
    order = np.argsort(-popularity)

    def rank(row: int, n: int):
        ranked = order[order != row][:n]
        return ranked, popularity[ranked]

    return rank


def _random_ranker(n_games: int):
    rng = np.random.default_rng(0)

    def rank(row: int, n: int):
        ranked = rng.choice(n_games - 1, size=n, replace=False)
        ranked[ranked >= row] += 1  # never return the query itself
        return ranked, np.zeros(n)

    return rank
