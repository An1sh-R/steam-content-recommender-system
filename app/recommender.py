"""The recommendation engine. This is the heart of the project.

Read this file top to bottom and you have the whole algorithm:

    1. Describe every game three ways: its tags, its genres, its description.
    2. Build a separate TF-IDF model for each of the three.
    3. To recommend, measure similarity in all three and blend the results.
    4. Rerank so well-reviewed games rise and shovelware sinks.
    5. Fetch the details and explain each pick.

Three models instead of one is the main design decision here. A description is
hundreds of words and a tag list is a dozen, so putting them in one document
lets the description drown out everything else. Keeping them apart also means we
can tell the user *which* of the three matched.
"""

import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app import config, database, explain

NON_LETTER_OR_DIGIT = re.compile(r"[^a-z0-9]+")


# --- Step 1: turn games into text ----------------------------------------


def build_documents(games):
    """Give each game three documents: one of tags, one of genres, one of prose.

    Returns three pandas Series, in the same order as `games`.
    """
    tag_documents = games["tags"].map(join_terms)

    # Genres are broad but reliable ("RPG"), and categories describe how a game
    # is played ("Single-player", "Co-op"). Both are short controlled lists, so
    # they belong together in one small space.
    genre_documents = (games["genres"] + games["categories"]).map(join_terms)

    description_documents = games["description"].str.lower()

    return tag_documents, genre_documents, description_documents


def join_terms(values):
    """["Turn-Based Strategy", "Open World"] -> "turn_based_strategy open_world"

    Each tag becomes a single token. If we left the spaces in, "Turn-Based
    Strategy" would be indexed as three unrelated words and would match any game
    that happens to mention strategy.
    """
    terms = []
    for value in values:
        term = NON_LETTER_OR_DIGIT.sub("_", value.lower()).strip("_")
        terms.append(term)
    return " ".join(terms)


# --- Step 2: build the three TF-IDF models -------------------------------


def build_tfidf_matrices(games):
    """Fit one TF-IDF model per document type. Returns three sparse matrices.

    Every row is one game and every column is one term. TfidfVectorizer scales
    each row to length 1, which is what lets us use a plain dot product as the
    cosine similarity later on.
    """
    tag_documents, genre_documents, description_documents = build_documents(games)

    # Tags and genres are hand-picked vocabularies where every term means
    # something, so we keep all of them and use no stopword list.
    tag_matrix = TfidfVectorizer().fit_transform(tag_documents)
    genre_matrix = TfidfVectorizer().fit_transform(genre_documents)

    # Descriptions are marketing prose, so we drop English stopwords, cap the
    # vocabulary, and use sublinear_tf -- a word used 20 times should not count
    # 20 times as much as a word used once.
    description_matrix = TfidfVectorizer(
        stop_words="english",
        max_features=30_000,
        sublinear_tf=True,
    ).fit_transform(description_documents)

    return tag_matrix, genre_matrix, description_matrix


def save_matrices(tag_matrix, genre_matrix, description_matrix, appids, directory):
    """Save the matrices so the API does not have to refit them on every start."""
    directory.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(directory / "tfidf_tags.npz", tag_matrix)
    sparse.save_npz(directory / "tfidf_genres.npz", genre_matrix)
    sparse.save_npz(directory / "tfidf_description.npz", description_matrix)
    np.save(directory / "appids.npy", appids)


def load_matrices(directory):
    """Load what save_matrices wrote."""
    if not (directory / "appids.npy").exists():
        raise FileNotFoundError(
            f"No matrices in {directory}. Build them first: python -m app.build --sample"
        )
    tag_matrix = sparse.load_npz(directory / "tfidf_tags.npz")
    genre_matrix = sparse.load_npz(directory / "tfidf_genres.npz")
    description_matrix = sparse.load_npz(directory / "tfidf_description.npz")
    appids = np.load(directory / "appids.npy")
    return tag_matrix, genre_matrix, description_matrix, appids


# --- Step 3: measure and blend similarity --------------------------------


def cosine_similarity_to_all(matrix, query_row):
    """How similar is every game to the game at `query_row`? Returns one score
    per game, between 0 and 1.

    The rows are already scaled to length 1, so multiplying two of them together
    gives the cosine similarity directly -- no division needed.
    """
    query_vector = matrix[query_row]
    similarities = matrix @ query_vector.T
    return np.asarray(similarities.todense()).ravel()


def combine_scores(
    tag_scores,
    genre_scores,
    description_scores,
    tag_weight=config.TAG_WEIGHT,
    genre_weight=config.GENRE_WEIGHT,
    description_weight=config.DESCRIPTION_WEIGHT,
):
    """Blend the three similarity scores into one score per game."""
    return (
        tag_weight * tag_scores
        + genre_weight * genre_scores
        + description_weight * description_scores
    )


def top_rows(scores, count):
    """The row numbers of the `count` highest scores, best first."""
    # There are only so many other games to choose from.
    count = min(count, len(scores) - 1)

    # argpartition pulls out the top `count` without bothering to sort the rest.
    # Sorting all 56,000 scores when we only want 300 would be much slower.
    best_unsorted = np.argpartition(-scores, count - 1)[:count]
    return best_unsorted[np.argsort(-scores[best_unsorted])]


# --- Step 4: rerank on quality -------------------------------------------


def apply_quality_boost(similarity_scores, popularity_scores):
    """Nudge well-reviewed games up and badly reviewed games down.

    The multiplier runs from QUALITY_FLOOR (0.70) up to 1.0, so a great game
    keeps all of its similarity and a terrible one keeps 70% of it. Quality can
    reshuffle games that are similarly relevant, but it can never push an
    irrelevant game above a relevant one -- which is the whole point of a floor.
    """
    floor = config.QUALITY_FLOOR
    multiplier = floor + (1 - floor) * popularity_scores
    return similarity_scores * multiplier


# --- Step 5: put it all together -----------------------------------------


class Engine:
    """Everything needed to answer recommendation requests, loaded once.

    Loading the matrices takes a few seconds, so the API builds one Engine at
    startup and reuses it for every request.
    """

    def __init__(self, tag_matrix, genre_matrix, description_matrix, appids, connection):
        self.tag_matrix = tag_matrix
        self.genre_matrix = genre_matrix
        self.description_matrix = description_matrix
        self.appids = appids
        self.connection = connection

        # The matrices are indexed by row number, but everything else in the app
        # talks in AppIDs, so we need a way to get from one to the other.
        self.row_of_appid = {}
        for row_number, appid in enumerate(appids):
            self.row_of_appid[int(appid)] = row_number

        # Small enough to keep in memory, and needed on every single request.
        self.popularity = database.load_popularity(connection, appids)
        self.tag_usage_counts = database.count_tag_usage(connection)

    @classmethod
    def load(cls):
        """Build an Engine from the files that app/build.py wrote."""
        tag_matrix, genre_matrix, description_matrix, appids = load_matrices(
            config.ARTIFACTS_DIR
        )
        connection = database.connect(config.DATABASE)
        return cls(tag_matrix, genre_matrix, description_matrix, appids, connection)

    def knows(self, appid):
        """Can we recommend games for this AppID?"""
        return appid in self.row_of_appid


def recommend(engine, appid, count=12):
    """Find the games most similar to `appid`, best first.

    Returns a list of game dictionaries, each with its similarity score, its
    final score, the breakdown across the three models, and a few short reasons.
    """
    query_row = engine.row_of_appid[appid]

    # 1. How similar is every game to this one, in each of the three spaces?
    tag_scores = cosine_similarity_to_all(engine.tag_matrix, query_row)
    genre_scores = cosine_similarity_to_all(engine.genre_matrix, query_row)
    description_scores = cosine_similarity_to_all(engine.description_matrix, query_row)

    # 2. Blend them into one similarity score per game.
    similarity_scores = combine_scores(tag_scores, genre_scores, description_scores)

    # A game is always a perfect match for itself, so take it out of the running.
    similarity_scores[query_row] = -np.inf

    # 3. Narrow down to a few hundred candidates worth reranking.
    candidate_rows = top_rows(similarity_scores, config.CANDIDATE_COUNT)

    # 4. Rerank those candidates on how well reviewed they are.
    final_scores = apply_quality_boost(
        similarity_scores[candidate_rows], engine.popularity[candidate_rows]
    )

    # 5. Keep the best `count` after reranking.
    best_positions = np.argsort(-final_scores)[:count]

    # 6. Collect the scores we want to show, keyed by AppID.
    scores_by_appid = {}
    recommended_appids = []
    for position in best_positions:
        row = candidate_rows[position]
        recommended_appid = int(engine.appids[row])
        recommended_appids.append(recommended_appid)
        scores_by_appid[recommended_appid] = {
            "similarity": float(similarity_scores[row]),
            "score": float(final_scores[position]),
            "parts": {
                "tags": float(tag_scores[row]),
                "genres": float(genre_scores[row]),
                "description": float(description_scores[row]),
            },
        }

    # 7. Look up the details and explain every pick.
    query_game = database.get_games(engine.connection, [appid])[0]
    recommended_games = database.get_games(engine.connection, recommended_appids)

    for game in recommended_games:
        game.update(scores_by_appid[game["appid"]])
        game["reasons"] = explain.explain_recommendation(
            query_game, game, game["parts"], engine.tag_usage_counts
        )

    return recommended_games
