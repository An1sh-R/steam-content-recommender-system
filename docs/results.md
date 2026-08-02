# Evaluation results

_55,973 games · 500 queries · full dataset · regenerate with `python -m app.evaluate`_

Each game's tags are split in half. Models are built on one half and
judged on the other, which they never see, so they cannot mark their own
homework. The ten games sharing the most hidden tags with a query count
as the right answers.

## Ranking quality

| model | notes | Precision@10 | Recall@50 | MAP@10 | NDCG@10 |
|---|---|---|---|---|---|
| rerank (shipped) | tuned weights + quality reranking | 0.024 | 0.057 | 0.014 | **0.261** |
| weighted tuned | the shipped weights | 0.024 | 0.055 | 0.013 | **0.257** |
| weighted description-heavy | t0.30 g0.20 d0.50 | 0.024 | 0.054 | 0.013 | **0.257** |
| weighted tag-heavy | t0.60 g0.15 d0.25 | 0.024 | 0.052 | 0.015 | **0.248** |
| weighted no description | t0.80 g0.20 d0.00 | 0.023 | 0.050 | 0.014 | **0.240** |
| tags only | single model | 0.020 | 0.042 | 0.012 | **0.220** |
| description only | single model | 0.012 | 0.026 | 0.006 | **0.177** |
| single model | everything in one TF-IDF model | 0.011 | 0.025 | 0.006 | **0.168** |
| genres only | single model | 0.005 | 0.018 | 0.002 | **0.158** |
| popularity | same list for every query | 0.000 | 0.002 | 0.000 | **0.078** |
| random | sanity floor | 0.000 | 0.001 | 0.000 | **0.060** |

## Beyond accuracy

| model | unique@10 | diversity@10 | novelty | poor@10 | tie rate | same publisher | p50 |
|---|---|---|---|---|---|---|---|
| rerank (shipped) | 91.1% | 0.811 | 0.346 | 13.0% | 0.000 | 8.3% | 51 ms |
| weighted tuned | 92.6% | 0.807 | 0.500 | 29.7% | 0.001 | 8.3% | 55 ms |
| weighted description-heavy | 92.4% | 0.806 | 0.499 | 29.7% | 0.001 | 9.0% | 56 ms |
| weighted tag-heavy | 92.5% | 0.816 | 0.504 | 29.5% | 0.000 | 5.7% | 57 ms |
| weighted no description | 92.6% | 0.819 | 0.503 | 29.2% | 0.032 | 4.2% | 54 ms |
| tags only | 92.4% | 0.830 | 0.499 | 29.3% | 0.111 | 2.6% | 53 ms |
| description only | 90.0% | 0.851 | 0.515 | 30.5% | 0.006 | 9.0% | 51 ms |
| single model | 87.0% | 0.858 | 0.516 | 31.5% | 0.001 | 7.3% | 51 ms |
| genres only | 93.8% | 0.841 | 0.476 | 25.9% | 0.663 | 3.6% | 57 ms |
| popularity | 0.2% | 0.916 | 0.000 | 0.0% | 0.000 | 0.0% | 7 ms |
| random | 0.4% | 0.935 | 0.477 | 17.6% | 0.980 | 0.0% | 0 ms |

## How to read this

- **Popularity is the bar.** A recommender that cannot beat one fixed list for every query is not worth its complexity.
- **`single model` vs `weighted tuned`** is the three-models-or-one question, measured rather than argued about.
- **`poor@10`** is the share of the page rated below 70% positive. It is the only metric here that can justify the quality reranking, because tag overlap is blind to whether a game is any good.
- **`tie rate`** catches a model that cannot tell games apart, and so is letting something else silently do the ranking.
- **Precision, Recall and MAP are all small, and that is expected.** Only ten games in the whole catalogue count as relevant for a query, and they are whichever titles share the most hidden tags -- often obscure games with nearly identical tag lists rather than ones a human would pick. Compare models against each other, not against 1.0.
- **NDCG is the headline** because it uses the whole graded overlap score instead of cutting the answer set off at ten.

**Self-recommendations: 0** across every model and query. A game must never recommend itself; this also catches duplicate re-releases slipping through.
