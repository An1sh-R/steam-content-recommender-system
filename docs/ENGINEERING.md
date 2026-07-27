# Engineering notes

The technical companion to the [README](../README.md): the decisions behind the
recommender, the experiments that settled them, and the ones that measurement
reversed.

Three documents, three jobs:

| document | purpose |
|---|---|
| [`README.md`](../README.md) | What the project is and how to run it |
| **this file** | Why it is built this way, and what was measured |
| [`CLAUDE.md`](../CLAUDE.md) | Exhaustive developer guide and source of truth |

Raw numbers live in [`evaluation/results.md`](../evaluation/results.md), which
`python -m evaluation.run_eval` regenerates.

---

## Contents

- [Why content-based, not collaborative filtering](#why-content-based-not-collaborative-filtering)
- [Design decisions](#design-decisions)
  - [The dataset header is malformed](#the-dataset-header-is-malformed--and-v1-shipped-on-it)
  - [Popularity is a Wilson lower bound](#popularity-is-a-wilson-lower-bound-not-owners)
  - [56k of 125,855 games](#56k-of-125855-games)
  - [Developers and publishers excluded from similarity](#developers-and-publishers-are-excluded-from-similarity)
  - [Reranking is multiplicative](#reranking-is-multiplicative)
- [Evaluation](#evaluation)
  - [The held-out tag protocol](#the-held-out-tag-protocol)
  - [Three TF-IDF spaces vs one](#three-tf-idf-spaces-vs-one)
  - [Final feature weights](#final-feature-weights)
  - [Quality reranking](#quality-reranking)
  - [The MMR prototype, and why it was removed](#the-mmr-prototype-and-why-it-was-removed)
- [Performance](#performance)
- [Limitations](#limitations)
- [Future work](#future-work)

---

## Why content-based, not collaborative filtering

**Because the honest answer to "should this be collaborative filtering?" is no.**

Collaborative filtering needs user–item interactions. This dataset has none, and
every public Steam interaction dataset is from 2016–2018 — it would cover roughly
4% of a 2026 catalogue, and joining it would mean entity resolution across two
sources with no shared key.

Rather than fake that with synthetic interactions — which would validate a
content model against data generated *by* a content model — the system is built
as what the data actually supports: a content-based retrieval system, evaluated
as an IR system.

Classical methods are also the point. TF-IDF and cosine similarity produce a
**score breakdown you can read**, which is what makes the explanations possible
at all. An embedding model would be a black box wrapped around the same task.

---

## Design decisions

### The dataset header is malformed — and V1 shipped on it

The published CSV declares **39 column names** but every data row has **40
fields**: header field 7 is `DiscountDLC count`, a missing comma. Every column
from index 7 onward is labelled with its neighbour's name.

| Header claims | Actually holds |
|---|---|
| `About the game` | `DLC count` (an integer) |
| `Positive` | `User score` (0–100) |
| `Categories` | `Publishers` |
| `Genres` | `Categories` |
| `Tags` | `Genres` |

A first version of this project shipped on the misaligned read. Its index was
built from *Categories + Publishers + Genres* — no tags, no descriptions — so it
was a **publisher matcher** that looked plausible because it silently grouped
games by publisher.

[`recommender/schema.py`](../recommender/schema.py) supplies the true layout
positionally. `tests/test_load.py` asserts on column *contents*, because a
misaligned read still produces a structurally valid DataFrame — only content
assertions catch it.

There is no naive read that works, which the notebook demonstrates executably:
`read_csv()` silently promotes AppID to the index and shifts every column before
the merge point; `index_col=False` shifts everything after it, turning
descriptions into DLC counts.

### Popularity is a Wilson lower bound, not owners

`Estimated owners` has 14 buckets with 60% of the catalogue in one of them — it
cannot rank anything. Steam reviews are **binary up/down votes**, so the correct
estimator is the Wilson score interval lower bound, not a star-rating formula
like IMDb's weighted rating, which assumes a continuous scale and an arbitrary
prior mean.

Sorting by raw positive ratio returns 1-review games at 100%. Wilson returns
*A Short Hike* (18,904 / 19,064). Wilson saturates above ~10k reviews, so reach
and recency are blended in:

```
popularity = 0.60 · wilson + 0.30 · log1p(reviews)/log1p(max) + 0.10 · recency
```

Recency uses an exponential decay with a 3-year half-life measured against
*today*, so rebuilding refreshes the front page. Unreleased games are clamped to
age 0 rather than scoring above 1.

### 56k of 125,855 games

Filter: `reviews ≥ 10` AND has tags AND `description ≥ 20 words` AND not a
playtest/demo/soundtrack, then reissues collapsed. **55,973 kept (44.5%).**

This is a *tag-coverage* filter, not a popularity filter: games with 0 reviews
have **0.9% tag coverage**; games with ≥1 review have **100%**. The excluded
third is unreleased and shovelware entries that are unrecommendable, not merely
unpopular.

Steam also lists some games under several AppIDs — Portal 2 and Assassin's
Creed 2 each appear multiple times with byte-identical descriptions, so they
ranked **first against themselves**. Collapsed on name + description + developer
(79 rows). The three-way key is deliberate: 349 rows share only a name and are
genuinely different games.

### Developers and publishers are excluded from similarity

They are used for explanations ("Also by Supergiant Games") and as an evaluation
diagnostic. V1 proved how strongly they dominate if included — and "same
publisher" is not a good reason to recommend a game.

The asymmetry is the point: a studio is a fair thing to *mention* once a game has
earned its place on content, and an unfair thing to rank on.

### Reranking is multiplicative

```
score = similarity × (0.70 + 0.30 · popularity)
```

Additive blending lets a popular-but-irrelevant game outrank a relevant one —
one term compensates for another. A multiplier treats quality as a *modifier on
relevance*: it reorders within a relevance band but cannot promote something
unrelated. The floor bounds the penalty.

### Explanations follow the ranking, not the game

A field is named only when it carried ≥ 25% of the combined score, largest
first. Descriptions carry the most weight, so a description-driven match must
not be explained as a tag match.

A *share* rather than an absolute threshold, because cosines are not comparable
across vocabularies of 449, 86 and 30,000 terms — but their contributions to one
score are.

Shared tags are ordered by **rarity**, the same reasoning as the IDF that
weighted them during retrieval: "both are Indie" explains nothing, "both are
Deck Building Roguelikes" explains the recommendation. *"Highly rated"* is backed
by the raw review ratio, deliberately not by the popularity score, which folds in
reach and recency and would make the sentence untrue.

---

## Evaluation

### The held-out tag protocol

There are no user interactions in this dataset, so there is no behavioural
ground truth. The system is evaluated as what it is — an IR system.

Each game's tags are split in half. The model is fitted on one half; relevance is
Jaccard overlap on the other. **The judging signal is never in the feature
space**, so the comparison is not circular — the trap most attribute-based
proxies fall into. The evaluation model is a separate fit from the production
model, which uses all tags.

Every metric has to change what happens next, or it is not reported:

| metric | answers |
|---|---|
| NDCG@10 | Is the ranking better? (headline) |
| Recall@50 | Is retrieval or ranking the limit? |
| unique@10 | Do we recycle the same few games? |
| diversity@10 | Are results near-duplicates? |
| novelty | Are we just showing blockbusters? |
| poor@10 | Are we recommending badly-reviewed games? |
| tie rate | Do scores actually discriminate? |
| same publisher | Have we regressed to V1? |
| self-retrieval | Does a game recommend itself? (must be 0) |

**Dropped for failing that bar:** MAP@10 (moves with NDCG) and Precision@10
(needs an arbitrary cut-off; at Jaccard ≥ 0.3 on half-sized tag sets it read
~0.08 for every model and ordered them identically to NDCG).

*All results below: 55,973 games · 500 stratified queries ·
[full tables](../evaluation/results.md)*

### Three TF-IDF spaces vs one

There are 451 distinct tags but ~28,000 distinct description words. In a single
shared space, prose dominates the vocabulary and distorts IDF for tags.

| model | NDCG@10 | vs single space |
|---|--:|--:|
| **weighted, three spaces** | **0.259** | **+53%** |
| tags only | 0.221 | +31% |
| description only | 0.180 | +7% |
| single concatenated space | 0.169 | — |
| genres only | 0.158 | −7% |
| popularity baseline | 0.077 | −54% |
| random | 0.075 | −56% |

Three spaces beat one concatenated document by **53%**, and the popularity
baseline by **3.4×**. The decisive argument is still explainability: three
*named* similarity numbers per candidate are what the explanations are written
from. It stayed cheap — roughly 15 lines more than a single document.

### Final feature weights

Swept, not guessed. **My stated prior was wrong**: I argued 0.60/0.15/0.25 from
field precision — 451 curated tags must beat 28,000 words of marketing prose.
The sweep disagreed.

| weights (tags / genres / desc) | NDCG@10 |
|---|--:|
| **0.35 / 0.20 / 0.45 (shipped)** | **0.259** |
| 0.30 / 0.20 / 0.50 | 0.258 |
| 0.60 / 0.15 / 0.25 *(my prior)* | 0.247 |
| 0.80 / 0.20 / — | 0.239 |

Validated on a fresh 800-query sample with a different seed: **+0.0093 NDCG,
95% CI [+0.0059, +0.0127]**. Descriptions evidently carry mechanics and setting
that tags miss.

The more useful finding is that the surface is **flat** — every sensible split
lands within ~5%, while collapsing to one space costs 46%. *Having* three spaces
matters far more than their ratio, so these numbers are not worth re-tuning.

### Quality reranking

| stage | NDCG@10 | poor@10 | novelty |
|---|--:|--:|--:|
| retrieval only | 0.259 | 28.7% | 0.497 |
| **+ quality rerank (shipped)** | **0.261** | **13.3%** | 0.347 |

`poor@10` is the share of a page rated below 70% positive. **Untouched retrieval
puts a badly-reviewed game in nearly 3 of every 10 slots.** The prior halves that
at no measurable ranking cost — paired bootstrap gives **+0.0011 NDCG, 95% CI
[−0.0012, +0.0034]**, indistinguishable from zero. The honest claim is
*neutral*, not *better*: it halves the bad-game rate for free.

`poor@10` was added to the harness *specifically because NDCG cannot see it*: tag
overlap is blind to whether a game is any good, so without it this stage had an
argument but no metric that could refute it.

The floor was swept too, and again the hand-picked 0.5 was too aggressive — it
cost NDCG *and* crushed novelty:

| floor | NDCG@10 | poor@10 | novelty |
|--:|--:|--:|--:|
| 1.00 (off) | 0.2588 | 28.7% | 0.497 |
| 0.85 | 0.2608 | 21.0% | 0.426 |
| **0.70 (shipped)** | **0.2609** | **13.3%** | **0.348** |
| 0.50 | 0.2580 | 6.2% | 0.256 |
| 0.00 | 0.2407 | — | 0.116 |

0.70 is the knee: the strongest prior that still costs nothing. The trade-off
that remains is real and stated — quality-weighting *is* a popularity bias, and
novelty drops 0.497 → 0.348. The floor is how much of one we chose to accept.

### The MMR prototype, and why it was removed

**MMR is not in this project. It was built, measured, and deleted.** Recorded
because the measurement is the useful artifact, and "we tried the textbook
component and it did not earn its place" is a decision, not an omission.

*Why it was implemented:* content retrieval clusters. Ask for games like a
franchise entry and the page fills with its own sequels, which carries less
information than its length suggests. MMR is the textbook fix.

```
value(i) = (1 − λ)·score(i) − λ·max_{j ∈ chosen} similarity(i, j)
```

*How it was evaluated:* as a stage in the same harness, swept over λ.

| λ | NDCG@10 | Δ | diversity@10 | Δ |
|--:|--:|--:|--:|--:|
| 0.00 | 0.2563 | — | 0.8136 | — |
| 0.15 | 0.2549 | −0.0014 | 0.8292 | +0.0155 |
| 0.30 | 0.2456 | −0.0107 | 0.8460 | +0.0324 |
| 0.40 | 0.2316 | −0.0247 | 0.8604 | +0.0468 |

*Why it was removed:* the aggregate numbers looked fine at λ=0.15 — cheap, and
statistically free. **The qualitative check killed it.** Against no
diversification at all:

| query | effect of MMR at λ=0.15 |
|---|---|
| Hades II | **identical output** |
| Stardew Valley | reorders, swaps 1 of 8 |
| Call of Duty | swaps 1 of 8 |
| Assassin's Creed Odyssey | swaps 2 positions — **still 6 AC games** |

It bought an aggregate diversity number no user would perceive, and left
untouched the franchise clustering that motivated it. Turning it up far enough to
break the clusters (λ≈0.35) filled the freed slots with noise — *YAR: Forgotten
Throne*, *MOOD* — because MMR's penalty distorts the relevance score itself.

The failure is **structural, not tuning**: sequels are similar to the query *and
to each other in the very space MMR penalises*, so no single λ separates
"redundant" from "on-topic". A global knob cannot express a query-specific
problem.

Cost of keeping it would have been ~50 lines, a 300×300 pairwise matrix per
request, an API parameter and a UI slider — for a benefit that is real in
aggregate and invisible in practice. `diversity@10` remains in the harness as a
**regression guard**.

A publisher cap *does* fix it and was prototyped: p90 crowding 4 → 2 and
same-publisher 9.0% → 4.8% for −0.0055 NDCG, filling freed slots with genuinely
similar games rather than noise. It is listed under future work rather than
shipped, because the pipeline is better off simple until the problem is worth a
mechanism.

---

## Performance

Measured on the full 55,973-game catalogue, p50 over 40 queries:

| endpoint | p50 | note |
|---|--:|---|
| `GET /recommend/{appid}` | **33 ms** | retrieve + rerank + hydrate + explain |
| — `engine.similar` alone | 24 ms | of which retrieval is 23 ms |
| `GET /games?q=` (typeahead) | 10 ms | flat, whatever you type |
| `GET /games/{appid}` | 3 ms | |
| `GET /popular` | 3 ms | indexed, narrow-then-hydrate |
| `GET /browse` (unfiltered) | 3 ms | |
| `GET /browse?genres=` | 60–219 ms | 219 ms is `Indie`, matching 40k of 56k rows |
| `GET /facets/genres` | 11 ms | |
| offline build | ~60 s | full dataset, one command |

Three optimisations were worth making. Each was found by measuring, not guessing:

**1. Cosine written `matrix @ query.T`, not `query @ matrix.T`.**
The natural-looking form transposes the full 56k × 30k description matrix on
every call. p50 dropped **280 ms → 25 ms** for bit-identical results.

**2. Narrow, then hydrate.**
Attaching tag/genre lists before `LIMIT` made SQLite build them for all 56k rows.
Selecting AppIDs first and hydrating just those: **99 ms → 2.5 ms**.

**3. A covering index and a one-character `+` for the typeahead.**
`search_names` was **0.1 ms for `"a"` and 1,070 ms for a title matching
nothing** — precisely backwards for a search box, since typing a specific title
is what a typeahead is *for*. The query plan explained it:

```
SCAN games USING INDEX idx_games_popularity
```

SQLite satisfied `ORDER BY popularity DESC` by walking the popularity index and
testing `LIKE` row by row until it had 20 matches. A common substring stops after
a few rows; a rare one walks all 55,973, each a random read into a table that
carries the descriptions.

| query | before | after |
|---|--:|--:|
| `a` | 0.1 ms | 6.7 ms |
| `portal` | 154 ms | 5.7 ms |
| `hades` | 243 ms | 5.8 ms |
| `zzzznotagame` (no match) | **1,070 ms** | **5.4 ms** |

Two changes: `ORDER BY +popularity DESC`, where the unary plus makes the ordering
non-indexable so SQLite filters first and sorts the survivors; and a covering
index on exactly the columns the typeahead reads, so the scan never touches the
180 MB table. Best case is now slower (0.1 → 6.7 ms) — worth it for a flat,
predictable 6 ms. The test asserts on the *query plan* rather than a latency
number, so the regression is caught deterministically.

No ANN index is warranted at this catalogue size, and the `Indie` browse is
deliberately left alone — it is imperceptible behind a cached UI, and fixing it
would cost a denormalised sort key for no benefit a user could feel.

---

## Limitations

Stated plainly, because a project that claims no weaknesses is not being honest
about its evaluation.

- **Tag overlap is a proxy for relevance, not human judgement.** It ranks systems
  reliably; it does not prove any individual recommendation is good. There are no
  pooled human relevance judgments here.
- **Franchise clustering is unaddressed.** Ask for games like Assassin's Creed
  Odyssey and you get six Assassin's Creed games. MMR was measured and did not
  fix it; a publisher cap would, and is scoped as future work.
- **`recall@50` is low everywhere (~0.06).** The "ideal top-10" under the
  protocol is whichever games share the most held-out tags — often obscure titles
  with near-identical tag sets rather than good recommendations. Read it as a
  *relative* signal between models, not an absolute miss rate.
- **Quality-weighting is a popularity bias.** Bounded and deliberate
  (novelty 0.497 → 0.347), but a bias.
- **The catalogue filter is itself a limitation.** Dropping 55% of games is
  defensible on tag coverage, but it means the recommender structurally cannot
  surface anything new or niche — and the evaluation, which scores only within
  the filtered catalogue, is blind to that by construction.
- **English-only** TF-IDF; non-English descriptions are handled poorly.
- **Cold catalogue.** Rebuilt offline from a static CSV; new Steam releases
  require a rebuild.
- **No personalisation.** By construction — there is no user model, only an
  item-to-item one.

---

## Future work

Scoped and justified, not a wishlist:

| | why |
|---|---|
| **Per-publisher cap in the top-10** | Directly fixes franchise clustering. Prototyped and measured: p90 crowding 4 → 2, same-publisher 9.0% → 4.8%, for −0.0055 NDCG. Must cap by *publisher*, not developer — the seven AC entries span five Ubisoft studios but one publisher. |
| **Pooled human relevance judgments** | The one thing that would upgrade the evaluation from proxy to ground truth. |
| **BM25 in the description space** | Length normalisation is a better fit for prose than raw TF-IDF; testable against the existing harness in an afternoon. |
| **Learned blend weights** | Logistic regression over the three field similarities, instead of a hand-swept grid. |
| **Query by multiple games** | Average several seed vectors — a small change to `retrieval.similar_rows`. |

### If this were production rather than a portfolio project

Deliberately not done here, because they would be complexity without payoff at
this scale:

1. **Structured logging and request tracing.** There is no observability at all.
   You cannot operate what you cannot see.
2. **Atomic artifact swaps.** Build to a versioned directory and flip a symlink,
   so a failed or concurrent build can never be observed half-applied.
3. **A startup consistency check.** Assert the `.npz` AppIDs match the catalogue
   and refuse to serve otherwise, rather than degrading silently.
4. **CI.** Tests, a sample build and a Docker build on every push.
5. **Pinned dependencies.** `>=` is right for a demo and wrong for
   reproducibility — a lockfile plus a pinned base image digest.
6. **An incremental build.** Rebuilding all 56k vectors to add one game is fine
   at this size and absurd at 10×.
7. **Online evaluation.** The held-out tag protocol ranks systems; it cannot tell
   you whether users click. Interleaving would make the offline proxy checkable
   against behaviour.
8. **Rate limiting and a response cache** on `/recommend`.
