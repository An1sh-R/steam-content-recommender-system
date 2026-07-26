# CLAUDE.md — Developer Guide

Single source of truth for this project. Read this before changing anything.

---

## 1. What this project is

A **content-based game recommender** over a Steam catalogue of ~56,000 games.
Given a game you like, it finds similar games, reranks them by community
quality, diversifies the list, and explains every recommendation.

It is a **portfolio and interview project**. It optimises for clarity,
explainability, and demonstrable engineering judgement — not for scale.

---

## 2. Philosophy

**Simplicity is a feature.** Given two approaches with similar expected
performance, take the simpler one. Complexity must earn its place with a
*measurable* benefit, demonstrated by the evaluation harness.

**Measure before tuning.** The evaluation harness is built before the
components it tunes. No weight in this project is chosen by taste; every one is
chosen by a number we can point at.

**Explainability is structural, not decorative.** The pipeline is arranged so
that "why was this recommended?" is answered by values it already computed.

**Readable over clever.** An interviewer should be able to read any module in
this repo in a few minutes and understand it completely. No factories, no
dependency-injection containers, no deep class hierarchies, no metaprogramming.

**Honesty about limits.** Where a method is a proxy or an approximation, the
README says so plainly.

---

## 3. Architecture

```
data/raw/games.csv          full dataset, 401 MB, NOT in git
data/sample/games_sample.csv  600 games, committed, runs out of the box
        │
        ▼   offline build  (make build)
recommender/                pure-Python library, no web framework imports
        │
        ├──► data/processed/catalogue.db     SQLite: metadata + facets
        └──► data/artifacts/*.npz            TF-IDF matrices + vectorizers
        │
        ▼   request path
api/                        FastAPI — thin HTTP layer, no business logic
        │
        ▼
app/                        Streamlit — presentation only
```

**The load-bearing boundary:** `recommender/` imports pandas, numpy, scipy and
scikit-learn — and nothing else. It never imports FastAPI or Streamlit. This is
what makes the whole pipeline testable and the evaluation harness possible.
Do not break it.

---

## 4. Modules and responsibilities

### `recommender/` — the library

| Module | Responsibility |
|---|---|
| `config.py` | Every tunable: paths, thresholds, weights. Nothing else. |
| `schema.py` | **The column contract.** True field order + validation. See §6.1. |
| `load.py` | Raw CSV → typed DataFrame. Fixes columns, coerces types. No filtering. |
| `clean.py` | Catalogue filter; splits multi-value fields into lists. |
| `popularity.py` | Wilson-based quality score. See §6.2. |
| `documents.py` | Builds per-field-group documents for vectorization. |
| `vectorize.py` | Fits/loads TF-IDF spaces; persists artifacts. |
| `retrieval.py` | Cosine similarity → candidates + per-field score breakdown. |
| `rerank.py` | Blends similarity with the quality prior. |
| `mmr.py` | MMR diversification. |
| `explain.py` | Score breakdown → short human reasons. |
| `catalogue.py` | SQLite: build the DB, serve faceted browse queries. |
| `engine.py` | Composes the pipeline. The public entry point. |
| `build.py` | Offline build CLI (`python -m recommender.build`). |

### `api/` — FastAPI

Routes call `Engine` and serialize. **No computation in routes.** If you find
yourself writing a loop in `routes.py`, it belongs in `recommender/`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + artifact version |
| `GET /games?q=&limit=` | Typeahead for the selectbox |
| `GET /games/{appid}` | Game detail |
| `GET /recommend/{appid}?k=&diversity=` | Primary workflow |
| `GET /popular?k=` | Landing page |
| `GET /browse?genres=&tags=&platform=&price_max=` | Faceted browse |

### `app/` — Streamlit

`main.py` (navigation + 3 modes), `api_client.py` (the only place `requests`
appears), `components.py` (game card, filter sidebar).

### `evaluation/`

`protocol.py` (held-out-tag split), `metrics.py`, `baselines.py`,
`run_eval.py` → writes `evaluation/results.md`.

---

## 5. Data flow

**Offline** (`python -m recommender.build`, ~1 min on the full dataset):

```
games.csv → schema (fix columns) → load (types) → clean (filter to ~56k)
          → popularity (Wilson) → documents → vectorize
          → data/artifacts/*.npz + data/processed/catalogue.db
```

The cleaned DataFrame is passed between build steps **in memory** and never
serialised. Nothing at serve time reads it — vectors come from `.npz`,
metadata from SQLite — so an intermediate file would be an artifact with no
consumer. Batch jobs that need it (the evaluation harness) call
`load()` + `clean()` themselves.

**Online** (per request):

```
Streamlit → FastAPI → Engine
                        ├─ retrieval : weighted cosine → top 300 + breakdown
                        ├─ rerank    : × quality prior
                        ├─ mmr       : greedy diversification → k
                        ├─ catalogue : hydrate metadata from SQLite
                        └─ explain   : short reasons
```

**Everything keys on AppID.** Titles are display-only — the dataset has 1,210
duplicate names. The selectbox shows `"Name (Year) — Developer"` and carries
AppID as the value.

---

## 6. Design decisions

### 6.1 The dataset header is malformed — this is the most important thing to know

The published CSV declares **39 column names** but every data row has **40
fields**. Header field 7 is `DiscountDLC count` — a missing comma between
`Discount` and `DLC count`. Every column from index 7 onward is therefore
labelled with its neighbour's name:

| Header claims | Actually holds |
|---|---|
| `About the game` | `DLC count` (an integer) |
| `Positive` | `User score` (0–100) |
| `Categories` | `Publishers` |
| `Genres` | `Categories` |
| `Tags` | `Genres` |

**Version 1 of this project shipped on the misaligned read.** Its TF-IDF index
was built from *Categories + Publishers + Genres* — no tags, no descriptions.
The result was a publisher-matcher that looked plausible (Witcher 3 → Cyberpunk
2077 → Gwent) because it was silently grouping games by publisher.

**Never read this CSV with its own header.** Always go through `load.load_raw()`,
which supplies `schema.RAW_COLUMNS` positionally. `tests/test_load.py` asserts
on column *contents*, because a misaligned read still produces a
structurally-valid DataFrame — only content assertions catch it.

### 6.2 Popularity uses the Wilson lower bound, not owners or raw ratio

`Estimated owners` has **14 distinct buckets** with 60% of the catalogue in one
of them. It cannot rank anything. It is kept for display and filtering only.

Steam reviews are **binary up/down votes**, so the correct estimator is the
Wilson score interval lower bound — not a star-rating formula like IMDb's
weighted rating, which assumes a continuous scale and an arbitrary prior mean.

Sorting by raw positive ratio returns 1-review games at 100%. Wilson returns
*A Short Hike* (18,904/19,064). Wilson saturates above ~10k reviews, so reach
and recency are added:

```
popularity = 0.60 · wilson + 0.30 · log1p(reviews)/log1p(max) + 0.10 · recency
```

Recency uses an exponential decay with a 3-year half-life, measured against
*today* rather than a fixed date — so rebuilding refreshes the front page. Games
with a future release date are clamped to age 0 rather than scoring above 1.

The resulting landing page (full catalogue): Black Myth: Wukong, Schedule I,
Baldur's Gate 3, Lethal Company, Satisfactory, Balatro, Vampire Survivors.

### 6.3 Catalogue filtered to ~56k of 125,855 games

Filter: `reviews ≥ 10` AND has tags AND `description ≥ 20 words` AND not a
playtest/demo/soundtrack, then reissues collapsed. **Measured: 55,973 of
125,855 games kept (44.5%).**

**Reissues.** Steam lists some games under several AppIDs — Portal 2,
Assassin's Creed 2 and BRINK each appear two or three times with byte-identical
descriptions. Being perfect content matches, they ranked *first against
themselves*. `clean._drop_reissues` collapses rows sharing name + description +
developer, keeping the most-reviewed (79 rows). The three-way key is deliberate:
349 rows share only a name and are genuinely different games.

Justification: games with 0 reviews have **0.9% tag coverage**; games with ≥1
review have **100%**. The missing third of the catalogue is unreleased and
shovelware entries that are unrecommendable, not merely unpopular. After
filtering: 100% tag coverage, 99.8% genres, median 60 reviews.

`MIN_REVIEWS` is in `config.py` — one line to revisit.

### 6.4 Field-weighted TF-IDF spaces, not one concatenated document

There are 451 distinct tags but ~28,000 distinct description words. In a single
shared space, prose dominates the vocabulary and distorts IDF for tags.

Three spaces, combined by weighted cosine:

```
similarity = 0.60·cos(tags) + 0.15·cos(genres+categories) + 0.25·cos(description)
```

The decisive argument is explainability: this yields three *named* similarity
numbers per candidate, so `explain.py` and the UI score breakdown fall out of
the architecture. It also keeps IDF within each vocabulary and makes each
field's contribution ablatable.

`sublinear_tf=True` on the description space — raw TF lets a word repeated 20×
in marketing copy carry 20× weight.

**Outcome (M3): it stayed cheap.** `documents.py` is 45 lines, `vectorize.py`
is 55, and the three spaces cost roughly 15 lines more than a single document
would have. Measured shapes: tags 449 terms, genres+categories 86, description
30,000. The breakdown it produces is already visible in `/recommend`.

### 6.4.1 The fitted vectorizers are not persisted

Recommendation is item-to-item over a fixed catalogue, so no text is ever
transformed at request time. Only the matrices and the AppID ordering are
needed, which drops a pickle-compatibility hazard and an artifact. If a future
feature needs to vectorise new text, refit at build time and save then.

### 6.4.2 Multi-word terms are slugged, not tokenised

"Turn-Based Strategy" becomes `turn_based_strategy`. Underscores are word
characters, so scikit-learn's default tokenizer keeps it as one term instead of
splitting it into three, without needing a custom analyzer.

### 6.5 Developers and Publishers are excluded from similarity

They are used for explanations ("Also by Supergiant Games") and as an
**evaluation diagnostic** (same-publisher rate in the top-10). V1 proves how
strongly they dominate if included — and "same publisher" is not a good reason
to recommend a game.

### 6.6 Reranking is multiplicative

```
score = similarity × (0.5 + 0.5 · popularity)
```

Additive blending (V1 used `0.6·sim + 0.2·pop + …`) lets a popular-but-
irrelevant game outrank a relevant one — one term compensates for another.
Multiplicative treats quality as a *modifier on relevance*: it reorders within
similar-relevance bands but cannot promote an irrelevant game. The `0.5 +`
floor bounds the penalty.

Keep the simplest version that works. The harness decides whether anything more
is warranted.

### 6.6.1 Cosine is written `matrix @ query.T`, not `query @ matrix.T`

The natural-looking form transposes the full 56k × 30k description matrix on
every call. Measured p50 dropped from **280 ms to 25 ms** end-to-end for
bit-identical results. One line, so it is worth having; nothing further about
retrieval is optimised, and no ANN index is warranted at this catalogue size.

### 6.7 Query exclusion is by AppID, never by rank

V1 dropped rank 0 assuming it was the query game. After quality weighting the
query often was not rank 0, so it leaked into its own results in **22% of
queries** while a legitimate recommendation was silently discarded. Exclude by
AppID and keep the regression test.

### 6.8 Explanations are short

One line each. `"Shares 9 tags"`, `"Similar genres"`, `"Similar gameplay
description"`, `"Highly rated by the community"`. Shared tags are selected by
*rarity* (highest IDF), because the rarest shared tag is the most informative.
`explain.py` returns structured data; formatting stays in the UI.

### 6.9 SQLite holds the catalogue, not the vectors

SQLite stores game metadata and indexed tag/genre/category facets — SQL is
genuinely the right tool for faceted browse filtering. TF-IDF matrices are
`.npz` on disk, loaded into memory once at startup.

The DB is a **read-only derived artifact**, rebuilt from scratch by
`recommender.build` and never written to at request time. Multi-value fields
are exploded into indexed child tables (`game_tags`, `game_genres`,
`game_categories`) rather than stored as delimited strings, so filtering can
use an index. `CHILD_TABLES` is the single source of truth: adding a facet
means adding one entry, and the tables, indexes and select-list follow.

### 6.10 Read paths narrow first, then hydrate

`get_games(con, appids)` is the only place rows are materialised. Anything that
needs a page of results selects AppIDs first and passes them in.

This is not stylistic. Attaching the tag/genre/category lists *before* applying
`LIMIT` made SQLite build them for all 56k rows: the unfiltered landing-page
query took **99 ms**. Selecting AppIDs first, plus indexes on the sortable
columns, brought it to **2.5 ms**.

Filtered browse ranges 11–203 ms depending on genre breadth (`Indie` matches
40k of 56k games and is the worst case). That is imperceptible in a Streamlit
UI and is deliberately left alone — optimising it would cost a denormalised
sort key for no benefit a user could feel.

---

## 7. Do NOT introduce

These were deliberately removed or rejected. Reintroducing any of them needs an
explicit decision recorded in this file.

| Not this | Why |
|---|---|
| **PostgreSQL** | Read-only catalogue, single writer, sub-100 MB. SQLite is sufficient and needs no infrastructure. |
| **Redis** | The expensive path is already fast. V1's cache key included every parameter, so hit rate was ~0, and it was a hard runtime dependency that 500'd the app when absent. |
| **Authentication / login / JWT / sessions / user accounts** | No ML value. V1's version was insecure (client-supplied `user_id`, no ownership check). |
| **Quiz / player profiles / game traits** | V1's hand-written 10-entry tag→trait map collapsed 112k games into 235 distinct vectors, with up to 4,376 games tied at max similarity. The quiz did not rank; popularity did. |
| **Collaborative filtering** | No user-item interactions in this dataset. Every public Steam interaction dataset is from 2016–2018 and would cover ~4% of a 2026 catalogue. |
| **Additional datasets** | One dataset, one join key (AppID), no entity resolution. |
| **Embeddings / transformers / LLMs / deep learning** | Classical, explainable methods are the point of this project. |
| **Cloud deployment (Render/Vercel/Supabase/Upstash)** | Docker locally is the deployment story. |
| **Synthetic interaction data** | Simulated users generated from content similarity would validate a content model circularly. |

---

## 8. Coding conventions

- **Python 3.11+.** `from __future__ import annotations` at the top of modules
  using annotations.
- **Type hints on public functions.** Not on every local variable.
- **Module docstring on every module** stating its one responsibility.
- **Comments explain *why*, not *what*.** If a line needs a "what" comment,
  rewrite the line.
- **Functions over classes.** Use a class only for genuine state (`Engine`).
- **No I/O at import time.** V1 opened a Redis connection and unpickled 40 MB at
  module import, which made it untestable. Load in an explicit function.
- **Pure functions in the library.** Pass data in, return data out. Side effects
  live in `build.py` and `catalogue.py`.
- **No linter or formatter.** Readable code and tests are the standard; keep
  lines around 100 characters and formatting consistent with what is already
  there. Deliberately one less dependency and one less config file.
- **Tests colocated by module name**: `recommender/clean.py` → `tests/test_clean.py`.
- Prefer `pathlib` over `os.path`; f-strings over `%` and `.format()`.

---

## 9. Development workflow

1. Work milestone by milestone (§11). Each milestone ends runnable.
2. Write the test alongside the module, not after the milestone.
3. Run `pytest` before considering a milestone done.
4. Anything that changes ranking behaviour → re-run the evaluation harness and
   report the delta.
5. Record non-obvious decisions in §6 of this file as they are made.

**Working against the sample.** `data/sample/games_sample.csv` (600 games,
2.5 MB) is committed. Everything runs against it with `--sample`, so a fresh
clone works without the 401 MB download. Regenerate it with `make sample`.

---

## 10. Common commands

Plain commands, no build tool — everything is one obvious line.

```bash
pip install -r requirements-dev.txt

pytest                                  # tests

python -m recommender.build --sample    # build against the committed 600-game sample
python -m recommender.build             # build against the full dataset
python scripts/make_sample.py           # regenerate the sample (needs the full dataset)
python -m evaluation.run_eval           # evaluation -> evaluation/results.md

uvicorn api.main:app --reload           # API
streamlit run app/main.py               # UI
docker compose up                       # both services
```

*(Commands past `make_sample.py` land in later milestones.)*

---

## 11. Roadmap

| Milestone | Contents | Status |
|---|---|---|
| **M0** | Foundation, column contract, sample dataset, test harness | ✅ done |
| **M1** | `clean.py` + SQLite catalogue + `build.py` | ✅ done |
| **M2** | Wilson popularity + first vertical slice (Popular page live) | ✅ done |
| **M3** | Documents, TF-IDF, retrieval, `/recommend` | ✅ done |
| **M4** | Evaluation harness + baselines *(before any tuning)* | next |
| **M5** | Rerank, MMR, explanations — tuned against M4 | |
| **M6** | Streamlit UI, three modes | |
| **M7** | Docker, README, `eda.ipynb` | |

### Future improvements (explicitly out of scope for now)

- Pooled human relevance judgments to supplement the tag-overlap proxy
- BM25 weighting as an alternative to TF-IDF in the description space
- Learned blend weights (logistic regression over the field similarities)
- Query-by-multiple-games (average several seed vectors)
- Tag co-occurrence analysis for a "related tags" browse affordance
