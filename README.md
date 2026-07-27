# Game Recommender System

A content-based Steam game recommender over ~56,000 games. Given a game you
like, it finds similar games, reranks them by community quality, and explains
every recommendation.

**Stack:** Python · scikit-learn · FastAPI · Streamlit · SQLite · Docker

> **Status: under active development (v2).** The application is being rebuilt
> from scratch. See [`CLAUDE.md`](CLAUDE.md) for the full design, architecture,
> and roadmap. This README is rewritten with evaluation results at M7.

---

## Quickstart

```bash
docker compose up          # → UI on :8501, API docs on :8000/docs
```

The image bakes in a 600-game sample catalogue, so a fresh clone runs with no
build step and no dataset download.

Or locally:

```bash
pip install -r requirements-dev.txt
python -m recommender.build --sample
uvicorn api.main:app --reload      # :8000
streamlit run app/main.py          # :8501
pytest
```

For the full catalogue, download the
[Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
to `data/raw/games.csv` (401 MB, not in git).

---

## Results

Measured on 55,973 games with 500 stratified queries. Ground truth is a
**held-out tag protocol** — each game's tags are split in half, the model is
fitted on one half, relevance is judged on the other, so the judging signal is
never in the feature space.

| model | NDCG@10 | tie rate | same publisher |
|---|--:|--:|--:|
| **weighted, three TF-IDF spaces** | **0.259** | 0.002 | 8.8% |
| tags only | 0.221 | 0.104 | 2.4% |
| description only | 0.180 | 0.009 | 9.2% |
| single concatenated space | 0.169 | 0.000 | 7.5% |
| popularity baseline | 0.077 | 0.000 | 0.0% |
| random | 0.075 | 0.980 | 0.0% |

Three weighted spaces beat one concatenated document by **53%** and the
popularity baseline by **3.4×**. Full table, including diversity, novelty and
coverage: [`evaluation/results.md`](evaluation/results.md).

Tag overlap is a proxy for relevance, not human judgement — it ranks systems
reliably, it does not prove any individual recommendation is good.

### What the quality prior buys

| stage | NDCG@10 | poor@10 |
|---|--:|--:|
| retrieval only | 0.259 | 28.7% |
| + quality rerank | 0.261 | **13.3%** |

`poor@10` is the share of a page rated below 70% positive. **Untouched
retrieval puts a badly-reviewed game in nearly 3 of every 10 slots**; the
quality prior halves that at no measurable cost in ranking quality (+0.0011
NDCG, 95% CI [−0.0012, +0.0034] — indistinguishable from zero). The strength
was swept, not guessed; my hand-picked value was too aggressive.

### What was removed after measuring it

**MMR is not in this project.** It was implemented, evaluated, and deleted in
the same milestone. It bought +0.019 diversity@10 for a statistically free
NDCG cost — but returned *output identical to no diversification* on half the
queries tried, and left untouched the franchise clustering that motivated it:
"games like Assassin's Creed Odyssey" still returned 6 Assassin's Creed games.
Turning it up far enough to break them replaced them with noise, because
sequels are similar to the query *and to each other* in the very space MMR
penalises.

Franchise clustering is therefore a **known limitation**. A publisher cap
solves it and is measured in §6.6.2 of [`CLAUDE.md`](CLAUDE.md); it is left as
future work rather than shipped, because the pipeline is better off simple.

## A note on the dataset

The published CSV has a malformed header: it declares **39 columns** while every
data row has **40 fields** (`DiscountDLC count` is two columns with a missing
comma). Reading it naively mislabels 32 of 40 fields — descriptions become DLC
counts, tags become genres, categories become publishers.

`recommender/schema.py` documents and fixes this; `tests/test_load.py` guards it.
See §6.1 of [`CLAUDE.md`](CLAUDE.md).

---

## Progress

- [x] **M0** — Foundation, column contract, sample dataset, tests
- [x] **M1** — Cleaning + SQLite catalogue (55,973 of 125,855 games kept)
- [x] **M2** — Wilson popularity + first vertical slice (API + UI + Docker)
- [x] **M3** — TF-IDF retrieval, `/recommend/{appid}` (~25 ms per query)
- [x] **M4** — Evaluation harness, baselines, weight sweep
- [x] **M5** — Quality reranking, explanations (~24 ms per query)
- [ ] **M6** — Streamlit UI
- [ ] **M7** — Docker, docs, EDA notebook
