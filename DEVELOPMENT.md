# Development guide

Single source of truth for this project. Read this before changing anything.

---

## 1. What this project is

A **content-based game recommender** over a Steam catalogue of ~56,000 games.
Given a game you like, it finds similar games, reranks them by community
quality, and explains every recommendation.

It is a **portfolio and interview project**. It optimises for clarity,
explainability, and demonstrable engineering judgement — not for scale.

### The document set

| document | audience | job |
|---|---|---|
| `README.md` | first-time visitor | What it is, how to run it. Skimmable in 5 minutes. |
| `docs/ENGINEERING.md` | interested engineer | Decisions, evaluation, experiments, trade-offs. |
| **`DEVELOPMENT.md`** (this file) | anyone changing the code | Exhaustive source of truth. |
| `evaluation/results.md` | — | Generated metric tables. Never hand-edited. |

Keep the README free of deep technical discussion; it belongs in
`docs/ENGINEERING.md`, and its full reasoning belongs in §6 below.

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
| `rerank.py` | Scales similarity by the quality prior. |
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
| `GET /recommend/{appid}?k=` | Primary workflow |
| `GET /popular?k=` | Landing page |
| `GET /browse?q=&genres=&platform=&max_price=&sort_by=` | Faceted browse |
| `GET /facets/{column}` | Filter options for the browse UI |

`/popular` is `/browse` with no filters. It is kept as a named endpoint because
"the front page" is a distinct thing to ask for, and it costs one line.

### `app/` — Streamlit

`main.py` (two modes: **Browse** and **Recommend similar games**),
`api_client.py` (the only place `requests` appears), `components.py`
(`game_card`, `game_grid`, `game_detail`).

Cover art is built from the AppID, never stored — see §6.11.

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
                        ├─ rerank    : × quality prior → top k
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
similarity = 0.35·cos(tags) + 0.20·cos(genres+categories) + 0.45·cos(description)
```

**Validated in M4 (NDCG@10, 55,973 games, 500 stratified queries):**

| | NDCG@10 |
|---|--:|
| weighted, three spaces | **0.259** |
| tags only | 0.221 |
| **single concatenated space** | **0.169** |
| popularity baseline | 0.077 |

Three spaces beat one concatenated document by **+53%**. That is the decision
justified, and it is now a measurement rather than an argument.

The decisive argument is explainability: this yields three *named* similarity
numbers per candidate, so `explain.py` and the UI score breakdown fall out of
the architecture. It also keeps IDF within each vocabulary and makes each
field's contribution ablatable.

`sublinear_tf=True` on the description space — raw TF lets a word repeated 20×
in marketing copy carry 20× weight.

**My prior about the weights was wrong.** I argued 0.60/0.15/0.25 from field
precision: 451 curated tags must beat 28,000 words of marketing prose. The
sweep says descriptions deserve *more* weight than tags (0.247 → 0.259 NDCG@10,
+0.0093 with a 95% bootstrap CI of [+0.0059, +0.0127] on a fresh 800-query
sample). Descriptions evidently carry mechanics and setting that tags miss.

The surface is **flat**: every sensible split lands within ~5%, while collapsing
to one space costs 46%. *Having* three spaces matters far more than their ratio,
so do not spend effort re-tuning these numbers.

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
score = similarity × (0.70 + 0.30 · popularity)
```

Additive blending (V1 used `0.6·sim + 0.2·pop + …`) lets a popular-but-
irrelevant game outrank a relevant one — one term compensates for another.
Multiplicative treats quality as a *modifier on relevance*: it reorders within
similar-relevance bands but cannot promote an irrelevant game. The floor bounds
the penalty.

**What it is for, measured (M5).** Without it, **28.7% of every recommendation
page is games rated below 70% positive**. NDCG cannot see this — tag overlap is
blind to whether a game is any good — so `poorly_rated_rate` was added to the
harness specifically to give this stage a metric that could refute it.

| floor | NDCG@10 | poor@10 | novelty |
|--:|--:|--:|--:|
| 1.00 (off) | 0.2588 | 28.7% | 0.497 |
| 0.85 | 0.2608 | 21.0% | 0.426 |
| **0.70** | **0.2609** | **13.3%** | **0.348** |
| 0.50 | 0.2580 | 6.2% | 0.256 |
| 0.00 | 0.2407 | — | 0.116 |

**0.70 is the knee**: the strongest prior that still costs nothing. On a fresh
800-query sample with a different seed the NDCG delta against no reranking is
**+0.0011, 95% CI [−0.0012, +0.0034]** — indistinguishable from zero. So the
honest claim is *neutral*, not *better*: it halves the bad-game rate for free.
Below 0.70 the ranking starts to pay.

**My hand-picked 0.5 was too aggressive** — same mistake as the M4 field
weights. It cut novelty to 0.256 (mean popularity percentile 0.74) for no gain
the harness could see. The trade-off that remains at 0.70 is real and stated:
novelty 0.497 → 0.348. Quality-weighting *is* a popularity bias; the floor is
how much of one we chose to accept.

### 6.6.2 MMR was built, measured, and removed

**There is no diversification stage.** MMR was implemented in M5, evaluated
against the harness, and deleted in the same milestone. It is recorded here
because the measurement is the useful artifact, and because "we tried the
textbook component and it did not earn its place" is a decision, not an
omission.

```
value(i) = (1 − d)·score(i) − d·max_{j ∈ chosen} similarity(i, j)
```

| d | NDCG@10 | Δ | diversity@10 | Δ |
|--:|--:|--:|--:|--:|
| 0.00 | 0.2563 | — | 0.8136 | — |
| 0.15 | 0.2549 | −0.0014 | 0.8292 | +0.0155 |
| 0.30 | 0.2456 | −0.0107 | 0.8460 | +0.0324 |
| 0.40 | 0.2316 | −0.0247 | 0.8604 | +0.0468 |

The numbers look acceptable at d=0.15 — cheap, statistically free. **The
qualitative check is what killed it.** Against no diversification at all:

| query | effect of MMR at d=0.15 |
|---|---|
| Hades II | **identical output** |
| Stardew Valley | reorders, swaps 1 of 8 |
| Call of Duty | swaps 1 of 8 |
| Assassin's Creed Odyssey | swaps 2 positions — **still 6 AC games** |

So it bought an aggregate diversity number that no user would perceive, and it
did not touch the franchise clustering that motivated adding it. Turning it up
far enough to break the clusters (d≈0.35) replaced them with *YAR: Forgotten
Throne* and *MOOD* — noise, because MMR's penalty distorts the score itself.

The failure is structural, not a tuning problem: sequels are similar to the
query **and to each other in the very space MMR penalises**, so no single λ
separates "redundant" from "on-topic". A global knob cannot express a
query-specific problem.

Cost of keeping it would have been ~50 lines, a 300×300 pairwise matrix per
request (+6 ms), a `retrieval.pairwise` helper, an API parameter and a UI
slider — all for a benefit that is real in aggregate and invisible in practice.
That is exactly the trade §2 says to refuse.

**What replaces it: nothing.** Franchise clustering is a known limitation, in
the README as such. A publisher cap ("at most 2 games per publisher") was
prototyped and does solve it — measured p90 crowding 4 → 2 and same-publisher
rate 9.0% → 4.8% for −0.0055 NDCG, filling freed slots with genuinely similar
games rather than noise. It is deliberately **not** shipped: the pipeline is
better off simple, and a real fix can be added later against the same
measurements. See future work.

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

One line each. `"Shares 12 tags including Souls-like and Dark Fantasy"`,
`"Similar genres"`, `"Similar gameplay description"`, `"Highly rated by the
community"`, `"Also by Supergiant Games"`.

**A field is named only when it drove the score.** `explain.py` converts the
per-space cosines into weighted contributions and mentions a space only if it
carried ≥ 25% of the combined score, largest first. This matters: descriptions
carry the most weight, so a description-driven match must not be explained as a
tag match. The reasons therefore *follow the ranking* instead of describing the
game in general — nothing is asserted that the model did not compute.

A share, not an absolute threshold, because cosines are not comparable across
vocabularies of 449, 86 and 30,000 terms — but their contributions to one score
are. It also means there is no magic number to calibrate per space.

**Shared tags are ordered by rarity**, from `catalogue.value_counts` — the same
reasoning as the IDF that weighted them during retrieval. "Both are Indie"
explains nothing; "both are Deck Building Roguelikes" explains the
recommendation.

`"Highly rated"` is backed by the **raw review ratio** (≥ 90% over ≥ 500
reviews), deliberately not by `popularity`, which also folds in reach and
recency and would make the sentence untrue.

Developers are excluded from *similarity* (§6.5) but are a fair *explanation*
once a game has earned its place on content — that asymmetry is the point.

### 6.8.1 Evaluation: the held-out tag protocol

There are no user interactions in this dataset, so there is no behavioural
ground truth. The system is evaluated as what it is — an IR system.

Each game's tags are split in half. The model under evaluation is fitted on one
half; relevance is Jaccard overlap on the other. **The judging signal is never
in the feature space**, so the comparison is not circular — the trap most
attribute-based proxies fall into. The evaluation model is a separate fit from
the production model, which uses all tags.

Metrics must be *actionable* — each one has to change what we do next:

| metric | answers |
|---|---|
| NDCG@10 | Is the ranking better? (headline) |
| Recall@50 | Is retrieval or ranking the limit? |
| unique@10 | Do we recycle the same few games? |
| diversity@10 | Are results near-duplicates? (regression guard; see §6.6.2) |
| novelty | Are we just showing blockbusters? |
| poor@10 | Are we recommending badly-reviewed games? (what rerank buys) |
| tie rate | Do scores actually discriminate? |
| same publisher | Have we regressed to V1? |
| self-retrieval | Does a game recommend itself? (must be 0) |

**Dropped for failing that bar:** MAP@10 (moves with NDCG) and Precision@10
(needs an arbitrary cut-off; at Jaccard ≥ 0.3 on half-sized tag sets it read
~0.08 for every model and ordered them identically to NDCG).

`recall@50` is low everywhere (~0.06). The "ideal top-10" is whichever games
share the most held-out tags, often obscure titles with near-identical tag sets
rather than good recommendations. Read it as a *relative* signal between models.

**Tag overlap is a proxy for relevance, not human judgement.** It ranks systems
reliably; it does not prove any single recommendation is good. Say so in the
README rather than implying more rigour than the protocol delivers.

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

### 6.10.1 The typeahead needed a covering index and a `+` (M7)

`search_names` was **0.1 ms for `"a"` and 1,070 ms for a title that matches
nothing**, which is the wrong way round: typing a specific title is what a
typeahead is *for*. The plan explained it —

```
SCAN games USING INDEX idx_games_popularity
```

— SQLite satisfied `ORDER BY popularity DESC` by walking the popularity index
and testing `LIKE` row by row until it had 20 matches. A common substring stops
after a few rows; a rare one walks all 55,973, each a random read into a table
that carries the descriptions.

Two changes, measured:

| query | before | after |
|---|--:|--:|
| `a` | 0.1 ms | 6.7 ms |
| `portal` | 154 ms | 5.7 ms |
| `hades` | 243 ms | 5.8 ms |
| `zzzznotagame` (no match) | **1,070 ms** | **5.4 ms** |

1. **`ORDER BY +popularity DESC`.** The unary plus makes the ordering
   non-indexable, so SQLite filters first and sorts the handful of survivors.
   It looks like a typo and is commented as load-bearing in `search_names`.
2. **A covering index** on exactly the columns the typeahead reads
   (`SEARCH_COLUMNS`). The scan is then served entirely from the index and never
   touches the 180 MB table.

Best case is now slower (0.1 → 6.7 ms) and the DB grew by a few MB. Both are
worth a flat, predictable 6 ms. `tests/test_catalogue.py` asserts on the *query
plan* rather than on a latency number, so the regression is caught
deterministically.

### 6.11 Cover art is derived from the AppID, not stored

```python
ARTWORK_URL = "https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"
```

The dataset ships a `Header image` column, and **it was removed from
`USED_COLUMNS` in M6.** Every value in it was already a Steam CDN URL that the
AppID reproduces, plus a `?t=` cache-busting timestamp that goes stale, and 23
rows were empty. Carrying it meant a column in the schema, a column in SQLite
and a field in the API response that only restated the primary key.

Deriving it instead means no image dataset, no asset pipeline, no storage, and
the 23 blanks fix themselves. Games with no artwork on Steam's CDN render a
broken-image placeholder rather than raising, so the card degrades to its text
— which is the graceful fallback we want, with no error handling to write.

This is the same rule that keeps `Recommendations` and `Metacritic score` out
of the catalogue: a column earns its place by feeding recommendation, browsing
or explanations. A URL you can compute feeds none of them.

### 6.12 Two modes, not three

The original plan had Popular, Similar and Browse as separate modes. Popular is
just Browse with no filters applied, so it is the **empty state of Browse**
rather than its own page. Fewer places to look, one less navigation decision,
and the cold-start answer is still the first thing a new user sees.

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

**Regenerating the README screenshots** is a rare manual task and deliberately
has no tooling checked in: run the UI, then drive it with Playwright installed
ad hoc (`pip install playwright && playwright install chromium`). A 150 MB
browser is not worth adding to a project whose dependency list is this short.

---

## 11. Roadmap

| Milestone | Contents | Status |
|---|---|---|
| **M0** | Foundation, column contract, sample dataset, test harness | ✅ done |
| **M1** | `clean.py` + SQLite catalogue + `build.py` | ✅ done |
| **M2** | Wilson popularity + first vertical slice (Popular page live) | ✅ done |
| **M3** | Documents, TF-IDF, retrieval, `/recommend` | ✅ done |
| **M4** | Evaluation harness + baselines *(before any tuning)* | ✅ done |
| **M5** | Rerank + explanations, tuned against M4 (MMR tried, removed) | ✅ done |
| **M6** | Streamlit UI, two modes | ✅ done |
| **M7** | README, architecture diagrams, packaging | ✅ done |

### Future improvements (explicitly out of scope for now)

- Pooled human relevance judgments to supplement the tag-overlap proxy
- BM25 weighting as an alternative to TF-IDF in the description space
- Learned blend weights (logistic regression over the field similarities)
- Query-by-multiple-games (average several seed vectors)
- Tag co-occurrence analysis for a "related tags" browse affordance
- A **publisher** cap in the top-10 to fix franchise clustering, the known
  limitation left open by removing MMR. Prototyped and measured in M5 (§6.6.2);
  not shipped, because the pipeline is better off simple until the problem is
  worth a mechanism. Note it must cap by *publisher*, not developer — the seven
  Assassin's Creed entries span five Ubisoft studios but one publisher.
