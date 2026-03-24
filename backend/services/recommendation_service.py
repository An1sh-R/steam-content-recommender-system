import json
import pickle
import redis
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from ml.content_recommender import recommend_games
from ml.player_profile import compute_player_profile
from backend.services.user_service import get_user_profile


# Redis connection
import os
redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True
)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

def get_data_path(*paths):
    return os.path.join(PROJECT_ROOT, "data", *paths)

# load game trait vectors once
with open(get_data_path("vectors", "game_traits.pkl"), "rb") as f:
    game_traits = pickle.load(f)


TRAIT_ORDER = [
    "exploration",
    "story",
    "challenge",
    "strategy",
    "social",
    "relaxation"
]


def profile_to_vector(profile: Dict[str, float]) -> np.ndarray:
    return np.array([profile[t] for t in TRAIT_ORDER])


def trait_similarity(player_vec: np.ndarray, game_vec: np.ndarray) -> float:
    return cosine_similarity(
        player_vec.reshape(1, -1),
        game_vec.reshape(1, -1)
    )[0][0]


def _resolve_player_profile(
    quiz_answers: Optional[List[int]],
    user_id: Optional[int],
    source: str = "game",
) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Determine which personalization mode to use for recommendations.
    - game: get_recommendations path
      - hybrid: when quiz_answers provided (and optionally saved profile for better blending)
      - game_only: no quiz answers
    - quiz: get_quiz_recommendations path
      - quiz_only: always use quiz answers (no saved profile blending)
    """

    if source == "quiz":
        # quiz-only mode: use quiz-derived profile and do not blend with saved profile.
        if quiz_answers:
            return compute_player_profile(quiz_answers), "quiz_only"
        return None, "quiz_only"

    # source == "game":
    if quiz_answers:
        # Game+quiz hybrid mode.
        if user_id:
            saved_profile = get_user_profile(user_id)
            if saved_profile:
                quiz_profile = compute_player_profile(quiz_answers)
                blended_profile = {}
                for trait in TRAIT_ORDER:
                    blended_profile[trait] = (
                        0.7 * quiz_profile[trait] +
                        0.3 * float(saved_profile[trait])
                    )
                return blended_profile, "hybrid"

        # No saved profile or no user ID: still consider as hybrid (game + quiz profile)
        return compute_player_profile(quiz_answers), "hybrid"

    # No quiz answers in game mode -> game-only.
    return None, "game_only"


def get_recommendations(
    game: str,
    quiz_answers: Optional[List[int]] = None,
    top_n: int = 5,
    user_id: Optional[int] = None,
) -> Dict:

    # Include user_id in cache key whenever available to avoid cross-user contamination.
    quiz_part = ",".join(map(str, quiz_answers)) if quiz_answers is not None else "none"
    user_part = str(user_id) if user_id is not None else "none"
    cache_key = f"recommendations:{game}:{top_n}:quiz={quiz_part}:user={user_part}"

    cached_result = redis_client.get(cache_key)

    if cached_result is not None:
        print("Cache hit")
        return json.loads(str(cached_result))

    print("Cache miss")

    # Tell Pylance what this is
    results: List[Dict] = recommend_games(game, top_n=50)

    player_profile, mode = _resolve_player_profile(quiz_answers=quiz_answers, user_id=user_id)
    player_vec: Optional[np.ndarray] = None
    if player_profile is not None:
        player_vec = profile_to_vector(player_profile)

    for r in results:

        idx = r["index"]   # index must come from content_recommender

        game_vec = np.array([game_traits[idx][t] for t in TRAIT_ORDER])

        # Keep ML core intact: only compute trait similarity when profile exists.
        if player_vec is not None:
            player_match = trait_similarity(player_vec, game_vec)
        else:
            player_match = 0.0

        r["player_match"] = float(player_match)

        r["hybrid_score"] = (
            0.6 * r["similarity_score"]
            + 0.2 * r["popularity"]
            + 0.2 * r["player_match"]
        )

    results = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)

    results = results[:top_n]

    payload = {"mode": mode, "recommendations": results}
    redis_client.set(cache_key, json.dumps(payload), ex=3600)

    return payload



GAME_METADATA = pd.read_csv("data/vectors/games_metadata.csv")
game_trait_matrix = np.array([list(trait_dict.values()) for trait_dict in game_traits])

def get_quiz_recommendations(
    quiz_answers: List[int],
    top_n: int = 5,
    metadata=GAME_METADATA,
    user_id: Optional[int] = None,
) -> Dict:
    user_part = str(user_id) if user_id is not None else "none"
    cache_key = f"quiz_recommendation:{top_n}:{','.join(map(str,quiz_answers))}:user={user_part}"
    cached_result = redis_client.get(cache_key)

    if cached_result is not None:
        print("Quiz Cache hit")
        return json.loads(str(cached_result))

    print("Quiz Cache miss")
    
    # Use same profile resolution logic for consistency
    player_profile, mode = _resolve_player_profile(
        quiz_answers=quiz_answers,
        user_id=user_id,
        source="quiz",
    )
    if player_profile is None:
        # No profile available, return empty results for quiz mode
        return {"mode": mode, "recommendations": []}
    
    player_vec = profile_to_vector(player_profile)

    # VECTORIZED similarity (FAST)
    similarities = cosine_similarity(
        player_vec.reshape(1, -1),
        game_trait_matrix
    ).flatten().astype(float)
    popularity = np.array(metadata["popularity"], dtype=float)
    # hybrid score
    hybrid_scores = 0.7 * similarities + 0.3 * popularity
    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        game_info = metadata.iloc[idx]

        results.append({
            "AppID": int(game_info["AppID"]),
            "Name": game_info["Name"],
            "Genres": game_info["Genres"],
            "Tags": game_info["Tags"],
            "popularity": float(popularity[idx]),
            "player_match": float(similarities[idx]),
            "hybrid_score": float(hybrid_scores[idx])
        })
    payload = {"mode": mode, "recommendations": results}
    redis_client.set(cache_key, json.dumps(payload), ex=3600)
    return payload