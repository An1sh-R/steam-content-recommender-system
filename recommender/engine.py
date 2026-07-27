"""Composes the pipeline and owns the loaded artifacts.

This is the library's public entry point. The API calls it; nothing here knows
about HTTP. The five stages -- retrieve, rerank, diversify, hydrate, explain --
are five lines in ``similar``; each one lives in its own module.
"""

from __future__ import annotations

import sqlite3

import numpy as np
from scipy import sparse

from recommender import catalogue, config, explain, mmr, rerank, retrieval, vectorize


class Engine:
    def __init__(
        self,
        matrices: dict[str, sparse.csr_matrix],
        appids: np.ndarray,
        connection: sqlite3.Connection,
    ) -> None:
        self.matrices = matrices
        self.appids = appids
        self.connection = connection
        self._row_of = {int(appid): row for row, appid in enumerate(appids)}

        # Small, read once, needed on every request. Both are derived from the
        # catalogue rather than stored as extra artifacts.
        self.popularity = catalogue.popularity_by_appid(connection, appids)
        self.tag_counts = catalogue.value_counts(connection, "tags")

    @classmethod
    def load(cls) -> Engine:
        matrices, appids = vectorize.load(config.ARTIFACTS_DIR)
        return cls(matrices, appids, catalogue.connect(config.CATALOGUE_DB))

    def knows(self, appid: int) -> bool:
        return appid in self._row_of

    def similar(
        self, appid: int, k: int = 12, diversity: float = config.DEFAULT_DIVERSITY
    ) -> list[dict]:
        """Games similar to ``appid``, best first."""
        rows, similarity, per_space = retrieval.similar_rows(self.matrices, self._row_of[appid])
        scores = rerank.apply(similarity, self.popularity[rows])
        picked = mmr.select(scores, retrieval.pairwise(self.matrices, rows), k, diversity)

        ranked = [int(self.appids[rows[i]]) for i in picked]
        # Keyed by AppID, not position: get_games drops anything it cannot find.
        detail = {
            int(self.appids[rows[i]]): {
                "similarity": float(similarity[i]),
                "score": float(scores[i]),
                "parts": {name: float(values[i]) for name, values in per_space.items()},
            }
            for i in picked
        }

        query, *games = catalogue.get_games(self.connection, [appid, *ranked])
        for game in games:
            game.update(detail[game["appid"]])
            game["reasons"] = explain.reasons(
                query, game, game["parts"], self.tag_counts
            )
        return games

    def popular(self, k: int = 24) -> list[dict]:
        return catalogue.browse(self.connection, limit=k)
