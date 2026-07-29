# Evaluation results

_55,973 games · 500 stratified queries · full dataset · regenerate with `python -m evaluation.run_eval`_

Ground truth is the **held-out tag protocol**: each game's tags are split in half,
the model is fitted on one half, relevance is judged as Jaccard overlap on the other.
The judging signal is never in the feature space, so the comparison is not circular.
It is a proxy for relevance, not human judgement.

## Ranking quality

| model | notes | NDCG@10 | Recall@50 |
|---|---|--:|--:|
| rerank (shipped) | quality prior | **0.261** | 0.059 |
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

| model | unique@10 | diversity@10 | novelty | poor@10 | tie rate | same publisher |
|---|--:|--:|--:|--:|--:|--:|
| rerank (shipped) | 92.0% | 0.814 | 0.347 | 13.3% | 0.000 | 8.5% |
| weighted tuned | 93.1% | 0.808 | 0.497 | 28.7% | 0.002 | 8.8% |
| weighted description-heavy | 93.4% | 0.807 | 0.497 | 28.9% | 0.002 | 9.6% |
| weighted tag-heavy | 93.2% | 0.819 | 0.498 | 28.9% | 0.001 | 5.5% |
| weighted no description | 93.3% | 0.822 | 0.501 | 29.0% | 0.029 | 4.0% |
| tags only | 92.7% | 0.835 | 0.496 | 28.8% | 0.104 | 2.4% |
| description only | 90.6% | 0.853 | 0.505 | 29.9% | 0.009 | 9.2% |
| single space | 89.8% | 0.861 | 0.509 | 30.6% | 0.000 | 7.5% |
| genres only | 94.1% | 0.845 | 0.484 | 26.2% | 0.676 | 3.8% |
| popularity | 0.2% | 0.916 | 0.000 | 0.0% | 0.000 | 0.0% |
| random | 95.8% | 0.923 | 0.511 | 29.9% | 0.980 | 0.0% |

## M5: what the quality prior costs, and what it buys

Same retrieval for both rows; `weighted tuned` is the prior off.
NDCG is the price. `poor@10` is the good.

| stage | notes | NDCG@10 | Δ | poor@10 | Δ | diversity@10 | Δ | novelty |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| weighted tuned | t0.35 / g0.20 / d0.45 | 0.259 | +0.000 | 28.7% | +0.0% | 0.808 | +0.000 | 0.497 |
| rerank (shipped) | quality prior | 0.261 | +0.002 | 13.3% | -15.4% | 0.814 | +0.005 | 0.347 |

## Integrity

- **Self-retrieval: 0** across all models and queries. A game must never recommend itself; this also guards the duplicate-reissue class of bug.
- Catalogue: 55,973 games. `unique@10` is distinct games returned divided by the 5,000 available top-10 slots.

## Reading this

- **Popularity is the bar.** A recommender that cannot beat one fixed list for every query is not earning its complexity.
- **`single space` vs `weighted tuned`** is the field-weighting decision, measured rather than argued.
- **`tie rate`** is the V1 post-mortem metric: its quiz collapsed 112k games into 235 distinct vectors, so popularity silently did all the ranking.
- **`recall@50` is low across the board.** The 'ideal top-10' is whichever games share the most held-out tags, which are often obscure titles with near-identical tag sets rather than good recommendations. Read it as a *relative* signal between models, not as an absolute miss rate.
- **`poor@10`** is the share of the page rated below 70% positive. NDCG cannot see it -- tag overlap is blind to whether a game is any good -- so it is the only metric that can justify the quality prior.
- **`diversity@10` no longer has a stage to justify.** MMR was built and measured against these metrics in M5, then deleted: it bought +0.019 diversity@10 but left franchise clustering untouched, and returned output identical to no diversification on half the queries tried. The metric stays as a regression guard. See DEVELOPMENT.md 6.6.2.
