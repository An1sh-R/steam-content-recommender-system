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
- [ ] **M4** — Evaluation harness
- [ ] **M5** — Reranking, MMR, explanations
- [ ] **M6** — Streamlit UI
- [ ] **M7** — Docker, docs, EDA notebook
