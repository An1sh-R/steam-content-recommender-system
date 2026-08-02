"""TF-IDF spaces: fit them at build time, load them at serve time.

Matrices are L2-normalised by TfidfVectorizer, so a dot product between two rows
is already their cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Tags and genres are controlled vocabularies: every term is meaningful, so no
# stopword list and no feature cap. Descriptions are prose, so cap the vocabulary
# and use sublinear TF -- a word repeated 20 times in marketing copy should not
# carry 20x the weight of one used once.
VECTORIZER_OPTIONS = {
    "tags": {},
    "genres": {},
    "description": {"stop_words": "english", "max_features": 30_000, "sublinear_tf": True},
}


def fit(documents: dict[str, pd.Series]) -> dict[str, sparse.csr_matrix]:
    return {
        name: TfidfVectorizer(**VECTORIZER_OPTIONS[name]).fit_transform(docs)
        for name, docs in documents.items()
    }  # pyright: ignore[reportReturnType]


def save(matrices: dict[str, sparse.csr_matrix], appids: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        sparse.save_npz(directory / f"tfidf_{name}.npz", matrix)
    np.save(directory / "appids.npy", appids)


def load(directory: Path) -> tuple[dict[str, sparse.csr_matrix], np.ndarray]:
    if not (directory / "appids.npy").exists():
        raise FileNotFoundError(
            f"No vectors in {directory}. Run: python -m recommender.build --sample"
        )
    matrices = {
        name: sparse.load_npz(directory / f"tfidf_{name}.npz") for name in VECTORIZER_OPTIONS
    }
    return matrices, np.load(directory / "appids.npy")
