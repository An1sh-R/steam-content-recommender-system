"""Composes the pipeline and owns the loaded artifacts.

This is the library's public entry point. The API calls it; nothing here knows
about HTTP. Reranking, MMR and explanations slot in at ``similar`` in M5.
"""

from __future__ import annotations

import sqlite3

import numpy as np
from scipy import sparse

from recommender import catalogue, config, retrieval, vectorize


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

    @classmethod
    def load(cls) -> Engine:
        matrices, appids = vectorize.load(config.ARTIFACTS_DIR)
        return cls(matrices, appids, catalogue.connect(config.CATALOGUE_DB))

    def knows(self, appid: int) -> bool:
        return appid in self._row_of

    def similar(self, appid: int, k: int = 12) -> list[dict]:
        """Games similar to ``appid``, most similar first."""
        rows, scores, per_space = retrieval.similar_rows(self.matrices, self._row_of[appid])
        ranked = [int(self.appids[row]) for row in rows[:k]]

        # Keyed by AppID, not position: get_games drops anything it cannot find.
        detail = {
            candidate: {
                "similarity": float(scores[i]),
                "parts": {name: float(values[i]) for name, values in per_space.items()},
            }
            for i, candidate in enumerate(ranked)
        }

        games = catalogue.get_games(self.connection, ranked)
        for game in games:
            game.update(detail[game["appid"]])
        return games

    def popular(self, k: int = 24) -> list[dict]:
        return catalogue.browse(self.connection, limit=k)
