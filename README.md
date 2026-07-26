# Game Recommender System

A content-based Steam game recommender over ~56,000 games. Given a game you
like, it finds similar games, reranks them by community quality, diversifies the
list, and explains every recommendation.

**Stack:** Python · scikit-learn · FastAPI · Streamlit · SQLite · Docker

> **Status: under active development (v2).** The application is being rebuilt
> from scratch. See [`CLAUDE.md`](CLAUDE.md) for the full design, architecture,
> and roadmap. This README is rewritten with evaluation results at M7.

---

## Quickstart

```bash
pip install -r requirements-dev.txt
pytest
```

Everything runs against `data/sample/games_sample.csv` (600 games, committed),
so a fresh clone works without downloading the full dataset.

For the full catalogue, download the
[Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
to `data/raw/games.csv` (401 MB, not in git).

---

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
- [x] **M1** — Cleaning + SQLite catalogue (56,052 of 125,855 games kept)
- [ ] **M2** — Wilson popularity + first vertical slice
- [ ] **M3** — TF-IDF retrieval
- [ ] **M4** — Evaluation harness
- [ ] **M5** — Reranking, MMR, explanations
- [ ] **M6** — Streamlit UI
- [ ] **M7** — Docker, docs, EDA notebook
