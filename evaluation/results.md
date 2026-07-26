# Evaluation results

_55,973 games · 500 stratified queries · full dataset · regenerate with `python -m evaluation.run_eval`_

Ground truth is the **held-out tag protocol**: each game's tags are split in half,
the model is fitted on one half, relevance is judged as Jaccard overlap on the other.
The judging signal is never in the feature space, so the comparison is not circular.
It is a proxy for relevance, not human judgement.

## Ranking quality

| model | notes | NDCG@10 | Recall@50 |
|---|---|--:|--:|
| weighted tuned | t0.35 / g0.20 / d0.45 | **0.259** | 0.057 |
| weighted description-heavy | t0.30 / g0.20 / d0.50 | **0.258** | 0.057 |
| weighted tag-heavy | t0.60 / g0.15 / d0.25 | **0.247** | 0.056 |
| weighted no description | t0.80 / g0.20 | **0.239** | 0.053 |
| tags only | single field | **0.221** | 0.043 |
| description only | single field | **0.180** | 0.029 |
| single space | everything concatenated | **0.169** | 0.028 |
| genres only | single field | **0.158** | 0.020 |
| popularity | same list for every query | **0.077** | 0.001 |
| random | sanity floor | **0.075** | 0.001 |

## Beyond accuracy

| model | unique@10 | diversity@10 | novelty | tie rate | same publisher | p50 |
|---|--:|--:|--:|--:|--:|--:|
| weighted tuned | 93.1% | 0.808 | 0.497 | 0.002 | 8.8% | 61 ms |
| weighted description-heavy | 93.4% | 0.807 | 0.497 | 0.002 | 9.6% | 21 ms |
| weighted tag-heavy | 93.2% | 0.819 | 0.498 | 0.001 | 5.5% | 62 ms |
| weighted no description | 93.3% | 0.822 | 0.501 | 0.029 | 4.0% | 6 ms |
| tags only | 92.7% | 0.835 | 0.496 | 0.104 | 2.4% | 7 ms |
| description only | 90.6% | 0.853 | 0.505 | 0.009 | 9.2% | 42 ms |
| single space | 89.8% | 0.861 | 0.509 | 0.000 | 7.5% | 22 ms |
| genres only | 94.1% | 0.845 | 0.484 | 0.676 | 3.8% | 4 ms |
| popularity | 0.2% | 0.916 | 0.000 | 0.000 | 0.0% | 0 ms |
| random | 95.8% | 0.923 | 0.511 | 0.980 | 0.0% | 0 ms |

## Integrity

- **Self-retrieval: 0** across all models and queries. A game must never recommend itself; this also guards the duplicate-reissue class of bug.
- Catalogue: 55,973 games. `unique@10` is distinct games returned divided by the 5,000 available top-10 slots.

## Reading this

- **Popularity is the bar.** A recommender that cannot beat one fixed list for every query is not earning its complexity.
- **`single space` vs `weighted tuned`** is the field-weighting decision, measured rather than argued.
- **`tie rate`** is the V1 post-mortem metric: its quiz collapsed 112k games into 235 distinct vectors, so popularity silently did all the ranking.
- **`recall@50` is low across the board.** The 'ideal top-10' is whichever games share the most held-out tags, which are often obscure titles with near-identical tag sets rather than good recommendations. Read it as a *relative* signal between models, not as an absolute miss rate.
- **Diversity@10** is what MMR (M5) must improve without costing NDCG.
