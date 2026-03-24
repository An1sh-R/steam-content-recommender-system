import pickle
import pandas as pd
import numpy as np
import difflib
from sklearn.metrics.pairwise import cosine_similarity

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

tfidf_path = os.path.join(PROJECT_ROOT, "data", "vectors", "tfidf_matrix.pkl")

with open(tfidf_path, "rb") as f:
    tfidf_matrix = pickle.load(f)

metadata = pd.read_csv("data/vectors/games_metadata.csv")

# build fast lookup
name_to_index = {name.lower(): idx for idx, name in enumerate(metadata["Name"])}

def find_game_index(game_name: str):

    game_name = game_name.lower()

    # exact match
    if game_name in name_to_index:
        return name_to_index[game_name]

    # substring match
    substring_matches = metadata[metadata["Name"].str.lower().str.contains(game_name, na=False)]

    if not substring_matches.empty:
        return substring_matches.index[0]

    # fuzzy fallback
    close_matches = difflib.get_close_matches(game_name,name_to_index.keys(),n=1,cutoff=0.8)

    if close_matches:
        return name_to_index[close_matches[0]]

    return None


def recommend_games(game_name: str, top_n: int = 5):

    idx = find_game_index(game_name)

    if idx is None:
        return []

    similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # normalize similarity
    similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())

    popularity_scores = np.array(metadata["popularity"].fillna(0).values)

    weighted_scores = (0.8 * similarities +0.2 * popularity_scores)

    similar_indices = weighted_scores.argsort()[::-1][1:top_n+1] # skip the first one since it's the same game

    # Include AppID so frontend can render Steam images/links reliably.
    results = metadata.iloc[similar_indices][["AppID", "Name", "Genres", "Tags", "popularity"]].copy()

    results["similarity_score"] = similarities[similar_indices]
    results["weighted_score"] = weighted_scores[similar_indices]
    results["index"] = similar_indices
    return results.to_dict(orient="records") # convert to list of dicts for easier JSON serialization