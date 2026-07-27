"""Run the evaluation and write evaluation/results.md.

    python -m evaluation.run_eval --sample     # quick, committed 600-game sample
    python -m evaluation.run_eval              # full catalogue
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import baselines, metrics, protocol
from recommender import build, config, load

RESULTS = Path(__file__).parent / "results.md"
N_QUERIES = 500
CANDIDATES = 50

# The weights under test. "tuned" is the shipped default from config.
WEIGHT_SWEEP = {
    "tuned": config.FIELD_WEIGHTS,
    "tag-heavy": {"tags": 0.60, "genres": 0.15, "description": 0.25},
    "description-heavy": {"tags": 0.30, "genres": 0.20, "description": 0.50},
    "no description": {"tags": 0.80, "genres": 0.20},
}

# The M5 stages, ablated against `weighted tuned` (which is this pipeline with
# both stages off). Each has to earn its place: rerank must not cost NDCG,
# MMR must buy diversity cheaply enough to be worth the loss.
STAGE_BASELINE = "weighted tuned"
STAGE_SWEEP = {
    "rerank": ("quality prior only", {"quality": True, "diversity": 0.0}),
    "rerank + mmr 0.15": ("shipped", {"quality": True, "diversity": config.DEFAULT_DIVERSITY}),
    "rerank + mmr 0.30": ("d=0.30", {"quality": True, "diversity": 0.30}),
    "rerank + mmr 0.50": ("d=0.50", {"quality": True, "diversity": 0.50}),
    "mmr 0.15 alone": ("no quality prior", {"quality": False, "diversity": 0.15}),
}


def evaluate(
    model: baselines.Model, queries, relevance, held_out, popularity_pct, publishers, ratio
):
    scores = {name: [] for name in ("ndcg", "recall", "diversity", "novelty", "poor", "tie")}
    publisher_rates, reached, latencies, self_hits = [], set(), [], 0

    for query in queries:
        start = time.perf_counter()
        ranked, ranking_scores = model.rank(int(query), CANDIDATES)
        latencies.append((time.perf_counter() - start) * 1000)

        truth = relevance.against_all(int(query))
        ranked_relevance = truth[ranked]

        scores["ndcg"].append(metrics.ndcg_at_k(ranked_relevance, truth))
        scores["recall"].append(metrics.recall_at_k(ranked, truth))
        scores["diversity"].append(metrics.intra_list_diversity(ranked, held_out))
        scores["novelty"].append(metrics.novelty(ranked, popularity_pct))
        scores["poor"].append(metrics.poorly_rated_rate(ranked, ratio))
        scores["tie"].append(metrics.tie_rate(ranking_scores))

        rate = metrics.same_publisher_rate(ranked, publishers, int(query))
        if rate is not None:
            publisher_rates.append(rate)

        reached.update(ranked[:10].tolist())
        self_hits += int(query in set(ranked[:10].tolist()))

    latencies.sort()
    return {
        "model": model.name,
        "note": model.note,
        **{name: float(np.mean(values)) for name, values in scores.items()},
        # Fraction of the top-10 slots filled by a distinct game. 1.0 means no
        # game was ever repeated across queries; popularity scores ~0 because it
        # returns one fixed list. Reported as a ratio, not a raw count, since
        # only len(queries) * 10 items can possibly be reached.
        "unique": len(reached) / (len(queries) * 10),
        "publisher": float(np.mean(publisher_rates)) if publisher_rates else 0.0,
        "self": self_hits,
        "p50": latencies[len(latencies) // 2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--queries", type=int, default=N_QUERIES)
    args = parser.parse_args()

    source = config.SAMPLE_CSV if args.sample else config.RAW_CSV
    print(f"loading {source.name} ...")
    games = build.prepare(load.load_raw(source))

    evaluated, held_out = protocol.prepare(games)
    relevance = protocol.Relevance(held_out)
    queries = protocol.sample_queries(games, args.queries)
    print(f"{len(games):,} games, {len(queries)} stratified queries\n")

    popularity_pct = games["popularity"].rank(pct=True).to_numpy()
    publishers = games["publishers"].to_numpy()
    ratio = (games["positive"] / games["total_reviews"]).to_numpy()

    rows = []
    for model in baselines.build(evaluated, WEIGHT_SWEEP, STAGE_SWEEP):
        started = time.perf_counter()
        rows.append(
            evaluate(model, queries, relevance, held_out, popularity_pct, publishers, ratio)
        )
        print(f"  {model.name:22s} ndcg={rows[-1]['ndcg']:.3f}  ({time.perf_counter()-started:.0f}s)")

    _write(pd.DataFrame(rows), games, queries, args.sample)
    print(f"\nwrote {RESULTS}")


def _stage_section(table: pd.DataFrame) -> list[str]:
    """Rerank and MMR against the same retrieval, as deltas.

    Absolute numbers cannot settle these two: the tag-overlap proxy is blind to
    whether a game is worth playing, and diversity is a cost in NDCG by
    construction. What it can do is *bound the price*, which is the decision.
    """
    rows = table.set_index("model")
    wanted = [STAGE_BASELINE, *(name for name in STAGE_SWEEP if name in rows.index)]
    if STAGE_BASELINE not in rows.index:
        return []

    base = rows.loc[STAGE_BASELINE]
    lines = [
        "",
        "## M5 stages: what rerank and MMR cost, and what they buy",
        "",
        f"Same retrieval for every row; `{STAGE_BASELINE}` is both stages off.",
        "NDCG is the price. `poor@10` and `diversity@10` are the goods.",
        "",
        "| stage | notes | NDCG@10 | Δ | poor@10 | Δ | diversity@10 | Δ | novelty |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name in wanted:
        r = rows.loc[name]
        lines.append(
            f"| {name} | {r['note']} | {r['ndcg']:.3f} | {r['ndcg'] - base['ndcg']:+.3f} | "
            f"{r['poor']:.1%} | {r['poor'] - base['poor']:+.1%} | "
            f"{r['diversity']:.3f} | {r['diversity'] - base['diversity']:+.3f} | "
            f"{r['novelty']:.3f} |"
        )
    return lines


def _write(table: pd.DataFrame, games, queries, sample: bool) -> None:
    table = table.sort_values("ndcg", ascending=False)
    catalogue = len(games)

    lines = [
        "# Evaluation results",
        "",
        f"_{catalogue:,} games · {len(queries)} stratified queries · "
        f"{'sample' if sample else 'full'} dataset · regenerate with "
        "`python -m evaluation.run_eval`_",
        "",
        "Ground truth is the **held-out tag protocol**: each game's tags are split in half,",
        "the model is fitted on one half, relevance is judged as Jaccard overlap on the other.",
        "The judging signal is never in the feature space, so the comparison is not circular.",
        "It is a proxy for relevance, not human judgement.",
        "",
        "## Ranking quality",
        "",
        "| model | notes | NDCG@10 | Recall@50 |",
        "|---|---|--:|--:|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['model']} | {r['note']} | **{r['ndcg']:.3f}** | {r['recall']:.3f} |"
        )

    lines += [
        "",
        "## Beyond accuracy",
        "",
        "| model | unique@10 | diversity@10 | novelty | poor@10 | tie rate | same publisher |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['model']} | {r['unique']:.1%} | {r['diversity']:.3f} | {r['novelty']:.3f} | "
            f"{r['poor']:.1%} | {r['tie']:.3f} | {r['publisher']:.1%} |"
        )

    lines += _stage_section(table)

    lines += [
        "",
        "## Integrity",
        "",
        f"- **Self-retrieval: {int(table['self'].sum())}** across all models and queries. "
        "A game must never recommend itself; this also guards the duplicate-reissue "
        "class of bug.",
        f"- Catalogue: {catalogue:,} games. `unique@10` is distinct games returned "
        f"divided by the {len(queries) * 10:,} available top-10 slots.",
        "",
        "## Reading this",
        "",
        "- **Popularity is the bar.** A recommender that cannot beat one fixed list "
        "for every query is not earning its complexity.",
        "- **`single space` vs `weighted tuned`** is the field-weighting decision, "
        "measured rather than argued.",
        "- **`tie rate`** is the V1 post-mortem metric: its quiz collapsed 112k games "
        "into 235 distinct vectors, so popularity silently did all the ranking.",
        "- **`recall@50` is low across the board.** The 'ideal top-10' is whichever "
        "games share the most held-out tags, which are often obscure titles with "
        "near-identical tag sets rather than good recommendations. Read it as a "
        "*relative* signal between models, not as an absolute miss rate.",
        "- **`poor@10`** is the share of the page rated below 70% positive. NDCG "
        "cannot see it -- tag overlap is blind to whether a game is any good -- "
        "so it is the only metric that can justify the quality prior.",
        "- **MMR's limit is visible in the numbers.** It buys diversity cheaply "
        "at d=0.15 and expensively after ~0.20. It does *not* break up franchise "
        "runs: sequels are similar to the query and to each other in the same "
        "content space MMR penalises, so any d strong enough to reject them also "
        "rejects the on-topic recommendations. See CLAUDE.md 6.6.2.",
    ]
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
