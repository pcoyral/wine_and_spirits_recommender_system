# S1 — Defence notes

*Study notes for the oral defence, not thesis text. S1 produces no evaluation
numbers. The few data facts below come from `results/eda/summary.csv` (named
next to them) or from the `make demo` output. If a file changes, the file is
right and these notes are out of date (CLAUDE.md rule 3). Decisions and
their alternatives are in `DECISIONS.md` (S1 entries, 21 Oct 2026).*

---

## The 60-second version

> "I built the measuring tools first, then the first two models. The metrics
> module defines how every model will be judged. Accuracy is measured by
> whether the model finds the wine a query was written from (HR@5, NDCG@5).
> Long-tail exposure is measured by how widely its recommendations spread over
> the catalogue (coverage, tail share, Gini, novelty). Diversity is how varied
> each list is. Every metric is tested on tiny examples I can calculate by
> hand, and all definitions were fixed before any results existed. All models
> share one interface, so they differ only in how they score a wine.
> Popularity is the baseline: it gives everyone the same wines from the
> most-reviewed wineries. Semantic compares the meaning of the user's words
> with the tasting notes using Sentence-BERT."

---

## 1. The shared interface — "Why one interface?"

```
fit(catalogue, holdout_ids)  →  learn whatever the model needs, ignoring test wines
score(query)                 →  one number per wine: how well it matches
recommend(query, k, filters) →  shared: score, filter, sort, top k
```

- **A fair comparison.** Filtering and ranking are the same code for every
  model, so a difference in results can only come from scoring. If each model
  ranked its own way, a result could be an artefact of sorting or filtering.
- **Deterministic.** When scores tie, the lower `wine_id` wins, so the same
  query always gives the same list (CLAUDE.md rule 1).
- **Filters** (max/min price, country, variety): a wine without a price never
  passes a price filter. Its price is unknown, not cheap, which matches the
  S0 decision not to make up prices.
- **Leakage guard.** `fit` receives the test IDs, and each model must ignore
  them when learning. The test wines stay in the catalogue, because they are
  the answers the evaluation looks for.

---

## 2. Popularity — the baseline

**What it does:** it ranks wines by how many reviews their winery has, and
ignores the query completely. Every user gets the same list.

**"Why measure popularity at winery level?"** Each wine in this dataset has
exactly one review, so wine-level popularity is the same for every wine.
Wineries are the smallest unit where popularity varies. It is also the unit
of the long-tail definition, so the baseline and the tail measure the same
thing.

**Ties:** score = winery review count + points / 1000. The whole number is the
review count, and the decimals are the points. Points are at most 100, so
points / 1000 is at most 0.1. They can break ties within a winery but can
never lift a wine above a winery with one more review.

**Leakage:** review counts are taken from the catalogue minus the test wines
(rule 4). The test proves it: hold out two of winery A's wines, and winery B
overtakes A (`tests/test_recommenders.py`).

**"Isn't this a straw man?"** No.

- **It is standard:** non-personalised popularity is the usual baseline in
  recommender research.
- **It is realistic:** it is roughly what a novice gets today, the
  "bestsellers" shelf or a well-known name.
- **It anchors the long-tail question:** SQ1 asks whether semantic matching
  shows more of the tail than popularity does. That needs a model that, by
  design, concentrates on the head.

You expect Popularity to score near zero on accuracy and on tail measures.
The question is how much Semantic improves on it.

**"Why not rank by points?"** Points measure a critic's opinion of quality,
not popularity, and they are unrelated to the tail definition. That was
option (b), recorded in `DECISIONS.md`. Option (c), one wine per winery, was
rejected because it adds a diversity rule to the baseline and would distort
the diversity comparison.

**A quirk to know:** the most-reviewed winery is a Portuguese producer,
"Wines & Winemakers" (`max_reviews_per_winery` in `summary.csv`; its name
shows in the `make demo` output). So Popularity recommends Portuguese wines
for every query, including "light crisp white". That is a good demo moment:
it shows why a novice needs a model that listens.

---

## 3. Semantic — the core model

**What Sentence-BERT is:** a neural network that turns a sentence into a list
of numbers (a vector). Sentences with similar meaning get vectors pointing in
similar directions. `all-MiniLM-L6-v2` is a small, fast version: 6 layers,
with each text becoming 384 numbers. It was trained on a very large set of
sentence pairs to put pairs with similar meaning close together. *Check the
model card before citing the exact training details.*

**How it ranks:**

1. Every tasting note is converted to a vector once, ahead of time.
2. The user's query is converted to a vector when they ask.
3. Each wine's score is the cosine similarity between the query vector and
   the wine's vector, i.e. how closely the two point in the same direction.
   1 means the same direction, 0 means unrelated.

**Why cosine:** it compares direction, not length, so a long note and a short
note with the same meaning score the same. All vectors are scaled to length 1,
which makes cosine a simple dot product: faster, and easier to explain.

**"Why should this work for novices?"** A keyword model like TF-IDF only
matches the exact words. "Not too heavy" shares no word with "light-bodied".
Sentence-BERT compares meaning, so it can bridge the vocabulary gap between a
novice's words and a critic's. That is exactly the main research question,
and the reason TF-IDF is a baseline.

**"Isn't there leakage? You embedded the test wines."** No. The model is
pretrained and learns nothing from this data, so there is nothing to fit. The
test wines must be embedded because they are the targets the queries have to
find. Fine-tuning the model on this data would have been leakage, and it is
OUT of scope anyway.

**The embedding cache:**

- **Why:** encoding about 120,000 notes takes minutes, so the result is stored
  on disk.
- **Fingerprint:** a hash of the model name, every `wine_id` and every note is
  stored with the cache. If anything changes, the fingerprint no longer
  matches and the cache is rebuilt. That prevents the silent risk of using
  embeddings from an old version of the data. The cache test checks this: a
  second run reuses the cache, and changing one note triggers a rebuild.

**Reproducibility:** pin the exact model version (`revision` in `config.yaml`)
together with the package versions. Embeddings can differ in the last
decimal places between machines, which practically never changes a ranking.

**Testing without the model:** the tests swap in a fake encoder that counts
three words, so the cosine similarities can be worked out by hand. The tests
then check the ranking logic and the cache with no download. The real model
is loaded only on first use.

---

## 4. The metrics — what each one answers

| Research question | Metrics |
|---|---|
| Main RQ: more relevant wines? | HR@5, NDCG@5 |
| SQ1: more of the long tail? | Catalogue coverage, tail share, Gini, novelty |
| Diversity (evaluation plan) | Intra-list diversity |

**HR@k (hit rate):** the share of queries whose target wine is in the top k.
Example: one hit out of two queries gives 0.5.

**NDCG@k:** like HR, but a hit near the top counts for more. A hit at rank r
earns 1/log₂(r+1): rank 1 = 1, rank 2 ≈ 0.63, rank 3 = 0.5, and a miss is 0.
With one correct wine per query, the best possible score is exactly 1, so no
extra normalisation step is needed.

**"Isn't one correct wine too strict?"** Yes, and it is a known limitation of
self-retrieval. Other wines may also fit the query but count as misses, so
absolute scores underestimate quality. That is why the sommelier check (P@5
on 30 queries) is in the evaluation plan. Self-retrieval is still fair for
comparing models, because every model is judged against the same targets.

**Catalogue coverage:** the share of the catalogue that appears in at least
one list. Popularity's top 5 is the same for everyone, so its coverage is tiny.

**Tail share:** the share of all recommendation slots filled by tail wines
(winery with ≤ 10 reviews). This is the most direct measure for SQ1.

**Gini of recommendations:** the same `gini()` function as the S0 catalogue
analysis. It counts how often each item is recommended, with
never-recommended items counting as 0.

- **Two hand examples:** the same two wines for everyone gives 0.5, and every
  wine recommended once gives 0.
- **Why both wine and winery level:** 200 queries × 5 slots against about
  120,000 wines leaves almost every wine at 0, so the wine-level Gini is close
  to 1 for every model. The winery level shows differences more clearly.
- **Decided before seeing results:** if you chose the level after seeing
  results, you could pick whichever made your model look better. Fixing it in
  advance removes that suspicion.

**Novelty:** the average "surprise" of a recommendation, −log₂(p), where p is
the wine's winery share of all reviews. Halving the share adds one bit, so a
wine from a winery with 1/8 of the reviews is worth 3 bits. It is measured at
winery level for the same reason as Popularity: every wine has one review.

**Intra-list diversity (ILD):** 1 minus the average cosine similarity between
every pair of wines in a list.

- Five near-identical wines score close to 0.
- Five very different wines score higher.
- Hand example: wines pointing the same way contribute 0, wines at right
  angles contribute 1.

**"Is that measure biased?"** It uses the SBERT embeddings for every model, so
one yardstick is applied to all. The honest weakness is that the yardstick is
the Semantic model's own view of similarity. Say so before the examiner does.

**"Why test on toy examples?"** Every metric gets an example small enough to
calculate by hand (`tests/test_metrics.py`, with the working in comments), so
the code and your understanding are checked against each other.

---

## 5. The demo — "What did you show your supervisor?"

- **What it does:** `make demo` runs your 5 casual queries through both
  models and saves the top 5 in `results/demo/demo_top5.md`.
- **What it isn't:** a qualitative look, not an evaluation. Five queries prove
  nothing statistically. The demo shows the models work and how differently
  they behave.
- **Your own words:** the queries are yours, written as a novice would say
  them, so they carry no expert vocabulary. They are not the 200 test
  queries, which come in S2.

---

## 6. Weaknesses to raise yourself

1. **One correct wine per query** underestimates absolute accuracy. The
   sommelier check covers that.
2. **ILD is measured in the Semantic model's own space,** which could favour
   it.
3. **MiniLM is a general English model,** not trained on wine. That is
   deliberate, because fine-tuning is OUT of scope. It is also the baseline a
   fine-tuned model could later be compared against.
4. **Popularity returns 5 wines from one winery.** That is correct for the
   baseline, but say so explicitly in the thesis.
5. **The wine-level Gini saturates near 1.** That is why the winery level is
   reported too.
6. **Open novelty issue:** a test wine whose winery has no other reviews gets
   p = 0. This must be settled before S2, for example by adding 1 to every
   winery's count (`DECISIONS.md`, open item).

---

## How the code is laid out

| Piece | One line you should be able to say |
|---|---|
| `Recommender.fit` | Stores the catalogue and remembers which wines to ignore when learning. |
| `Recommender.recommend` | Scores, filters, sorts (ties by `wine_id`) and returns the top k. Shared by every model. |
| `filter_mask` | Price, country and variety filters; unknown prices never pass a price filter. |
| `PopularityRecommender` | Winery review count without test wines, points as tie-breaker, same list for every query. |
| `SemanticRecommender` | SBERT vectors of length 1; score = dot product with the query vector = cosine. |
| `fingerprint`, `load_or_encode` | The embedding cache, rebuilt automatically if the model or notes change. |
| `hit_rate_at_k`, `ndcg_at_k` | Was the target found, and how high up? |
| `catalogue_coverage`, `tail_share` | How much of the catalogue, and how much of the tail, gets recommended? |
| `recommendation_gini` | How unequally recommendations are spread over items. |
| `novelty` | Average surprise in bits, from winery share. |
| `intra_list_diversity` | How varied each list is, in SBERT space. |
| `demo.py` | Top 5 of both models for your queries, saved for the supervisor. |
