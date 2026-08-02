"""Measuring how good the recommendations actually are.

    python -m app.evaluate --sample     # quick run on the committed sample
    python -m app.evaluate              # the full catalogue

The hard part of evaluating a recommender is deciding what "correct" means.
This dataset has no user history, so there is nothing to check our answers
against -- and grading a tag-based recommender on tag overlap would just be
marking its own homework.

So we split each game's tags in half. The models are built from one half, and we
judge them on the other half, which they never saw. It is still only a proxy for
whether a human would like the recommendation, but it is an honest one, and it
ranks models reliably.

Results are written to docs/results.md.
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app import build, config, recommender

RESULTS_FILE = config.ROOT / "docs" / "results.md"

DEFAULT_QUERY_COUNT = 500
RESULTS_PER_QUERY = 50
MIN_TAGS_TO_BE_A_QUERY = 8  # both halves need to be worth something

# The weight settings we compare, as (name, note, tag, genre, description).
WEIGHTED_MODELS = [
    ("tags only", "single model", 1.00, 0.00, 0.00),
    ("genres only", "single model", 0.00, 1.00, 0.00),
    ("description only", "single model", 0.00, 0.00, 1.00),
    ("weighted tuned", "the shipped weights",
     config.TAG_WEIGHT, config.GENRE_WEIGHT, config.DESCRIPTION_WEIGHT),
    ("weighted tag-heavy", "t0.60 g0.15 d0.25", 0.60, 0.15, 0.25),
    ("weighted description-heavy", "t0.30 g0.20 d0.50", 0.30, 0.20, 0.50),
    ("weighted no description", "t0.80 g0.20 d0.00", 0.80, 0.20, 0.00),
]


# --- Setting up the ground truth -----------------------------------------


def split_tags(tags):
    """Split a game's tags in half: even positions to train on, odd to judge on.

    Splitting by position rather than at random keeps the whole evaluation
    reproducible without threading a seed through everything.
    """
    return tags[0::2], tags[1::2]


def build_tag_matrix(tag_lists):
    """A games x tags matrix with a 1 wherever a game has a tag."""
    every_tag = set()
    for tags in tag_lists:
        every_tag.update(tags)

    column_of_tag = {}
    for column, tag in enumerate(sorted(every_tag)):
        column_of_tag[tag] = column

    rows = []
    columns = []
    for row, tags in enumerate(tag_lists):
        for tag in tags:
            rows.append(row)
            columns.append(column_of_tag[tag])

    values = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(tag_lists), len(column_of_tag))
    )


@dataclass
class EvaluationData:
    """Everything the metrics need, worked out once up front."""

    games_for_training: pd.DataFrame  # games with only their first half of tags
    held_out_tags: sparse.csr_matrix  # the second half, used as ground truth
    held_out_tag_counts: np.ndarray  # how many held-out tags each game has
    popularity_percentile: np.ndarray
    publishers: np.ndarray
    review_ratio: np.ndarray


def prepare_evaluation_data(games):
    """Hide half of every game's tags, and pre-compute what the metrics need."""
    training_tags = []
    held_out_tags = []
    for tags in games["tags"]:
        first_half, second_half = split_tags(tags)
        training_tags.append(first_half)
        held_out_tags.append(second_half)

    games_for_training = games.copy()
    games_for_training["tags"] = training_tags

    held_out_matrix = build_tag_matrix(held_out_tags)

    return EvaluationData(
        games_for_training=games_for_training,
        held_out_tags=held_out_matrix,
        held_out_tag_counts=np.asarray(held_out_matrix.sum(axis=1)).ravel(),
        popularity_percentile=games["popularity"].rank(pct=True).to_numpy(),
        publishers=games["publishers"].to_numpy(),
        review_ratio=(games["positive"] / games["total_reviews"]).to_numpy(),
    )


def true_relevance(data, query_row):
    """How relevant is every game to this query, judged on the hidden tags?

    This is the Jaccard overlap: shared tags divided by total distinct tags.
    Returns one score per game, between 0 and 1.
    """
    shared = data.held_out_tags @ data.held_out_tags[query_row].T
    shared = np.asarray(shared.todense()).ravel()

    total = data.held_out_tag_counts + data.held_out_tag_counts[query_row] - shared
    return np.divide(shared, total, out=np.zeros_like(shared), where=total > 0)


def ideal_top_ten(relevance):
    """The ten genuinely most relevant games. This is our 'correct answer' set."""
    return set(np.argsort(relevance)[::-1][:10].tolist())


def choose_query_games(games, how_many, seed=0):
    """Pick the games to test with, spread evenly across the popularity range.

    Without this the sample is almost all obscure titles, and the numbers stop
    describing the games anyone would actually search for.
    """
    eligible = games.index[games["tags"].map(len) >= MIN_TAGS_TO_BE_A_QUERY]
    if len(eligible) <= how_many:
        return np.asarray(eligible)

    deciles = pd.qcut(
        games.loc[eligible, "popularity"], 10, labels=False, duplicates="drop"
    )
    per_decile = max(1, how_many // (deciles.max() + 1))

    random = np.random.default_rng(seed)
    chosen = []
    for _, group in deciles.groupby(deciles):
        take = min(per_decile, len(group))
        chosen.append(random.choice(group.index, size=take, replace=False))

    return np.sort(np.concatenate(chosen))


# --- The metrics ---------------------------------------------------------


def precision_at_k(ranked_rows, relevant_set, k=10):
    """Of the k games we showed, what share were genuinely relevant?"""
    shown = ranked_rows[:k]
    hits = len(relevant_set & set(shown.tolist()))
    return hits / len(shown) if len(shown) else 0.0


def recall_at_k(ranked_rows, relevant_set, k=50):
    """Of the genuinely relevant games, what share did we find?

    Read together with precision this separates the two ways to fail: poor
    recall means we never retrieved the good games, while good recall with poor
    NDCG means we retrieved them but ranked them badly.
    """
    if not relevant_set:
        return 0.0
    found = len(relevant_set & set(ranked_rows[:k].tolist()))
    return found / len(relevant_set)


def average_precision(ranked_rows, relevant_set, k=10):
    """Precision measured each time we hit a relevant game, then averaged.

    Rewards putting the relevant games near the top rather than merely
    including them. Averaged over all queries this is MAP.
    """
    if not relevant_set:
        return 0.0

    hits = 0
    precision_total = 0.0
    for position, row in enumerate(ranked_rows[:k], start=1):
        if int(row) in relevant_set:
            hits += 1
            precision_total += hits / position

    return precision_total / min(len(relevant_set), k)


def ndcg_at_k(ranked_rows, relevance, k=10):
    """Like precision, but relevance is a score rather than yes/no, and getting
    the order right within the top k counts too.

    We divide by the best possible ranking, so 1.0 means perfect.
    """
    gains = relevance[ranked_rows[:k]]
    ideal_gains = np.sort(relevance)[::-1][:k]

    discount = np.log2(np.arange(2, len(gains) + 2))
    actual = float((gains / discount).sum())
    ideal = float((ideal_gains / np.log2(np.arange(2, len(ideal_gains) + 2))).sum())

    return actual / ideal if ideal > 0 else 0.0


def intra_list_diversity(ranked_rows, held_out_tags, k=10):
    """Are these k games all slight variations of each other?

    1 means completely different, 0 means identical. Measured on the held-out
    tags, so a list cannot look diverse just because we optimised for it.
    """
    rows = ranked_rows[:k]
    tag_vectors = held_out_tags[rows]

    lengths = np.sqrt(np.asarray(tag_vectors.multiply(tag_vectors).sum(axis=1)).ravel())
    lengths[lengths == 0] = 1.0

    similarity = np.asarray((tag_vectors @ tag_vectors.T).todense())
    similarity = similarity / np.outer(lengths, lengths)

    pairs = similarity[np.triu_indices(len(rows), k=1)]
    return float(1 - pairs.mean()) if pairs.size else 0.0


def novelty(ranked_rows, popularity_percentile, k=10):
    """Are we just listing the same blockbusters? Higher means more obscure."""
    return float(1 - popularity_percentile[ranked_rows[:k]].mean())


def poorly_rated_rate(ranked_rows, review_ratio, k=10, threshold=0.70):
    """What share of the page is games the community disliked?

    NDCG cannot see this at all -- tag overlap says nothing about whether a game
    is any good -- so this is the only metric that can justify the quality
    reranking step.
    """
    return float((review_ratio[ranked_rows[:k]] < threshold).mean())


def tie_rate(scores, k=50):
    """What share of the top k share a score with another result?

    A high tie rate means the model cannot actually tell these games apart, and
    whatever breaks the tie is silently doing the ranking.
    """
    top = scores[:k]
    if not len(top):
        return 0.0
    return float(1 - len(np.unique(top)) / len(top))


def same_publisher_rate(ranked_rows, publishers, query_row, k=10):
    """Have we accidentally built a publisher matcher instead of a recommender?"""
    if not publishers[query_row]:
        return None
    return float((publishers[ranked_rows[:k]] == publishers[query_row]).mean())


# --- The models we compare -----------------------------------------------


def rank_by_similarity(matrices, popularity, query_row, how_many,
                       tag_weight, genre_weight, description_weight,
                       use_quality_reranking=False):
    """Rank the catalogue for one query using the real engine functions."""
    tag_matrix, genre_matrix, description_matrix = matrices

    tag_scores = recommender.cosine_similarity_to_all(tag_matrix, query_row)
    genre_scores = recommender.cosine_similarity_to_all(genre_matrix, query_row)
    description_scores = recommender.cosine_similarity_to_all(description_matrix, query_row)

    scores = recommender.combine_scores(
        tag_scores,
        genre_scores,
        description_scores,
        tag_weight=tag_weight,
        genre_weight=genre_weight,
        description_weight=description_weight,
    )
    scores[query_row] = -np.inf

    if not use_quality_reranking:
        ranked_rows = recommender.top_rows(scores, how_many)
        return ranked_rows, scores[ranked_rows]

    # The shipped pipeline: widen to a few hundred candidates, then rerank.
    candidate_rows = recommender.top_rows(scores, config.CANDIDATE_COUNT)
    final_scores = recommender.apply_quality_boost(
        scores[candidate_rows], popularity[candidate_rows]
    )
    best = np.argsort(-final_scores)[:how_many]
    return candidate_rows[best], final_scores[best]


def rank_by_one_matrix(matrix, query_row, how_many):
    """Used for the 'everything in one TF-IDF model' comparison."""
    scores = recommender.cosine_similarity_to_all(matrix, query_row)
    scores[query_row] = -np.inf
    ranked_rows = recommender.top_rows(scores, how_many)
    return ranked_rows, scores[ranked_rows]


def rank_by_popularity(popularity, query_row, how_many):
    """The bar every recommender has to clear: the same list for every query."""
    order = np.argsort(-popularity)
    ranked_rows = order[order != query_row][:how_many]
    return ranked_rows, popularity[ranked_rows]


def rank_at_random(game_count, query_row, how_many, seed=0):
    """The sanity floor. Anything that cannot beat this is broken."""
    random = np.random.default_rng(seed)
    ranked_rows = random.choice(game_count - 1, size=how_many, replace=False)
    ranked_rows[ranked_rows >= query_row] += 1  # never pick the query itself
    return ranked_rows, np.zeros(how_many)


@dataclass
class Model:
    """One system we are comparing against the others."""

    name: str
    note: str
    kind: str  # "random", "popularity", "single model" or "weighted"
    tag_weight: float = 0.0
    genre_weight: float = 0.0
    description_weight: float = 0.0
    use_quality_reranking: bool = False


@dataclass
class FittedModels:
    """The matrices every model ranks with, fitted once and shared."""

    matrices: tuple
    single_matrix: sparse.csr_matrix
    popularity: np.ndarray
    game_count: int


def list_models():
    """Every system under test, in the order we report them."""
    models = [
        Model("random", "sanity floor", "random"),
        Model("popularity", "same list for every query", "popularity"),
        Model("single model", "everything in one TF-IDF model", "single model"),
    ]

    for name, note, tag_w, genre_w, description_w in WEIGHTED_MODELS:
        models.append(Model(name, note, "weighted", tag_w, genre_w, description_w))

    models.append(
        Model(
            "rerank (shipped)",
            "tuned weights + quality reranking",
            "weighted",
            config.TAG_WEIGHT,
            config.GENRE_WEIGHT,
            config.DESCRIPTION_WEIGHT,
            use_quality_reranking=True,
        )
    )
    return models


def fit_models(games):
    """Fit every TF-IDF model the comparison needs."""
    matrices = recommender.build_tfidf_matrices(games)

    # The design decision under test: three weighted models, or one big one?
    tag_docs, genre_docs, description_docs = recommender.build_documents(games)
    one_big_document = tag_docs + " " + genre_docs + " " + description_docs
    single_matrix = TfidfVectorizer(
        stop_words="english", max_features=30_000
    ).fit_transform(one_big_document)

    return FittedModels(
        matrices=matrices,
        single_matrix=single_matrix,
        popularity=games["popularity"].to_numpy(),
        game_count=len(games),
    )


def rank(model, fitted, query_row, how_many):
    """Rank the catalogue for one query, using whichever model was asked for."""
    if model.kind == "random":
        return rank_at_random(fitted.game_count, query_row, how_many)

    if model.kind == "popularity":
        return rank_by_popularity(fitted.popularity, query_row, how_many)

    if model.kind == "single model":
        return rank_by_one_matrix(fitted.single_matrix, query_row, how_many)

    return rank_by_similarity(
        fitted.matrices,
        fitted.popularity,
        query_row,
        how_many,
        tag_weight=model.tag_weight,
        genre_weight=model.genre_weight,
        description_weight=model.description_weight,
        use_quality_reranking=model.use_quality_reranking,
    )


# --- Running the whole thing ---------------------------------------------


def evaluate_model(model, fitted, query_rows, data):
    """Run one model over every query game and average its scores."""
    scores = {
        "precision": [], "recall": [], "map": [], "ndcg": [],
        "diversity": [], "novelty": [], "poor": [], "tie": [],
    }
    publisher_rates = []
    latencies = []
    games_ever_shown = set()
    self_recommendations = 0

    for query_row in query_rows:
        query_row = int(query_row)

        started = time.perf_counter()
        ranked_rows, ranking_scores = rank(model, fitted, query_row, RESULTS_PER_QUERY)
        latencies.append((time.perf_counter() - started) * 1000)

        relevance = true_relevance(data, query_row)
        relevant_set = ideal_top_ten(relevance)

        scores["precision"].append(precision_at_k(ranked_rows, relevant_set))
        scores["recall"].append(recall_at_k(ranked_rows, relevant_set))
        scores["map"].append(average_precision(ranked_rows, relevant_set))
        scores["ndcg"].append(ndcg_at_k(ranked_rows, relevance))
        scores["diversity"].append(intra_list_diversity(ranked_rows, data.held_out_tags))
        scores["novelty"].append(novelty(ranked_rows, data.popularity_percentile))
        scores["poor"].append(poorly_rated_rate(ranked_rows, data.review_ratio))
        scores["tie"].append(tie_rate(ranking_scores))

        rate = same_publisher_rate(ranked_rows, data.publishers, query_row)
        if rate is not None:
            publisher_rates.append(rate)

        top_ten = ranked_rows[:10].tolist()
        games_ever_shown.update(top_ten)
        if query_row in top_ten:
            self_recommendations += 1

    latencies.sort()
    result = {"model": model.name, "note": model.note}
    for metric, values in scores.items():
        result[metric] = float(np.mean(values))

    # How much of the catalogue we ever show. A model that returns the same list
    # every time scores near zero here.
    result["unique"] = len(games_ever_shown) / (len(query_rows) * 10)
    result["publisher"] = float(np.mean(publisher_rates)) if publisher_rates else 0.0
    result["self"] = self_recommendations
    result["p50_ms"] = latencies[len(latencies) // 2]
    return result


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_results(results, game_count, query_count, used_sample):
    """Write docs/results.md."""
    results = sorted(results, key=lambda r: r["ndcg"], reverse=True)
    dataset = "sample" if used_sample else "full"

    lines = [
        "# Evaluation results",
        "",
        f"_{game_count:,} games · {query_count} queries · {dataset} dataset · "
        "regenerate with `python -m app.evaluate`_",
        "",
        "Each game's tags are split in half. Models are built on one half and",
        "judged on the other, which they never see, so they cannot mark their own",
        "homework. The ten games sharing the most hidden tags with a query count",
        "as the right answers.",
        "",
        "## Ranking quality",
        "",
    ]
    lines += markdown_table(
        ["model", "notes", "Precision@10", "Recall@50", "MAP@10", "NDCG@10"],
        [
            [
                r["model"], r["note"],
                f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                f"{r['map']:.3f}", f"**{r['ndcg']:.3f}**",
            ]
            for r in results
        ],
    )

    lines += ["", "## Beyond accuracy", ""]
    lines += markdown_table(
        ["model", "unique@10", "diversity@10", "novelty", "poor@10", "tie rate",
         "same publisher", "p50"],
        [
            [
                r["model"], f"{r['unique']:.1%}", f"{r['diversity']:.3f}",
                f"{r['novelty']:.3f}", f"{r['poor']:.1%}", f"{r['tie']:.3f}",
                f"{r['publisher']:.1%}", f"{r['p50_ms']:.0f} ms",
            ]
            for r in results
        ],
    )

    total_self = sum(r["self"] for r in results)
    lines += [
        "",
        "## How to read this",
        "",
        "- **Popularity is the bar.** A recommender that cannot beat one fixed "
        "list for every query is not worth its complexity.",
        "- **`single model` vs `weighted tuned`** is the three-models-or-one "
        "question, measured rather than argued about.",
        "- **`poor@10`** is the share of the page rated below 70% positive. It is "
        "the only metric here that can justify the quality reranking, because "
        "tag overlap is blind to whether a game is any good.",
        "- **`tie rate`** catches a model that cannot tell games apart, and so is "
        "letting something else silently do the ranking.",
        "- **Precision, Recall and MAP are all small, and that is expected.** Only "
        "ten games in the whole catalogue count as relevant for a query, and they "
        "are whichever titles share the most hidden tags -- often obscure games "
        "with nearly identical tag lists rather than ones a human would pick. "
        "Compare models against each other, not against 1.0.",
        "- **NDCG is the headline** because it uses the whole graded overlap score "
        "instead of cutting the answer set off at ten.",
        "",
        f"**Self-recommendations: {total_self}** across every model and query. "
        "A game must never recommend itself; this also catches duplicate "
        "re-releases slipping through.",
        "",
    ]

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the recommender.")
    parser.add_argument("--sample", action="store_true", help="use the sample dataset")
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERY_COUNT)
    arguments = parser.parse_args()

    source = config.SAMPLE_CSV if arguments.sample else config.RAW_CSV
    print(f"Reading {source.name} ...")
    games = build.prepare_catalogue(build.load_raw_csv(source))

    data = prepare_evaluation_data(games)
    query_rows = choose_query_games(games, arguments.queries)
    print(f"{len(games):,} games, {len(query_rows)} queries\n")

    fitted = fit_models(data.games_for_training)

    results = []
    for model in list_models():
        started = time.perf_counter()
        result = evaluate_model(model, fitted, query_rows, data)
        results.append(result)
        elapsed = time.perf_counter() - started
        print(f"  {model.name:28s} ndcg={result['ndcg']:.3f}  ({elapsed:.0f}s)")

    write_results(results, len(games), len(query_rows), arguments.sample)
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
