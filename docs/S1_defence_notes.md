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
  S0 decision not to make up prices. A misspelled filter name is an error,
  never silently ignored.
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
- **Fingerprint:** a hash of the model name, the model version, every
  `wine_id` and every note is stored with the cache. If anything changes, the
  fingerprint no longer matches and the cache is rebuilt. That prevents the silent risk of using
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

**"What if there is nothing to measure?"** For example, a filter so strict
that no wine passes. Each metric then stops with an error instead of
returning "not a number" (nan), so an empty result can never end up in a
results table unnoticed.

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
6. **Novelty when p = 0:** a test wine whose winery has no other reviews
   would get p = 0. Resolved with add-one smoothing: every winery gets one
   extra review, so p > 0 and the shares still sum to 1 (`DECISIONS.md`).
   `novelty()` itself still refuses p = 0 with an error, as a safety net.
7. **The S1 code was reviewed** (21 Oct 2026). The fixes: the cache now also
   checks the model version, unknown filters are errors, metrics refuse
   empty input, and file handles are closed.

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

---

# Part 2 — TF-IDF, Attribute and `make eval`

*Same rule as above: no evaluation numbers exist yet; decisions and their
alternatives are in `DECISIONS.md`.*

## The 60-second version

> "S1 ends with four recommenders behind one interface, plus one command that
> evaluates them all. Each model adds one capability. Popularity ignores the
> user. Attribute understands structured wishes such as 'Italian, red, under
> $15'. TF-IDF matches the user's words with the critic's words. Semantic
> matches meaning. Comparing them on the same queries, with the same metrics,
> shows what each capability contributes: an *ablation*. The baselines use
> standard settings and are never tuned, and every model is fitted without the
> test wines. `make eval` refuses to run until the casual test queries exist,
> so no number exists that wasn't produced by the final protocol."

## 7. Why these four models — the ablation ladder

| Model | What it can understand | What comparing it with Semantic tests |
|---|---|---|
| Popularity | Nothing; same list for everyone | Does listening to the user help at all? |
| Attribute | Explicit facts: grape, country, colour, price | Is meaning more than a filter on facts? |
| TF-IDF | The exact words | Does matching meaning beat matching words? (the vocabulary gap) |
| Semantic | Meaning | The core model of the main RQ |

**"Why four and not just Semantic vs Popularity?"** Beating Popularity proves
very little, because any model that reads the query should beat it. TF-IDF is
the real test of the main claim. If Semantic only matched TF-IDF, the
vocabulary gap wouldn't matter. Attribute stands for what a wine-shop website
does: filters on facts.

## 8. TF-IDF — the keyword baseline

**How it works:** each tasting note becomes a vector with one entry per word
in the vocabulary.

- **TF (term frequency):** how often the word appears in the note, dampened
  (`sublinear_tf`): 1 + log(count), so a word repeated three times doesn't
  count three times as much.
- **IDF (inverse document frequency):** how rare the word is across all notes.
  A word that appears everywhere ("wine") weighs little; a rare word
  ("lychee") weighs a lot. scikit-learn computes ln((1 + n) / (1 + df)) + 1,
  where n is the number of notes and df the number containing the word.
- **Ranking:** the query becomes a vector the same way, and wines are ranked
  by cosine similarity. Vectors have length 1, so this is a dot product,
  exactly as in Semantic. The two models differ only in how they build
  vectors.

**Settings (A1, standard and not tuned):**

- **Single words:** no word pairs.
- **Stop words:** filler words like "and" and "the" are dropped, using
  scikit-learn's English list **minus 8 words that carry meaning**: "not",
  "no", "too", "very", "less", "more", "without", "full" (decision (b)). With
  the plain list, "not too heavy for a barbecue" became just "heavy barbecue",
  the opposite of the request, and "full-bodied" lost "full". Keeping them
  means the baseline is not weakened by a technicality.
- **`min_df = 2`:** a word must appear in at least 2 notes. That removes typos
  and one-off names.
- **"Why not tune it?"** Semantic isn't tuned either. Tuning only the
  baseline, or only your own model, would make the comparison unfair.

**Leakage:** the vocabulary and IDF are learned without the test wines. The
test shows the consequence: a word that appears only in a test wine
("lychee") doesn't exist for the model. The test wine is still recommendable
through its other words.

**Its built-in weakness, which is the point:** with no shared word there is
no match. "Citrus" scores 0 against a note saying "lemon"
(`test_tfidf_has_no_idea_of_meaning`). Keeping "not" doesn't give TF-IDF an
understanding of negation either: it only matches notes that also contain
the word "not".

## 9. Attribute — the structured baseline

**How it works:** a rule-based reader finds four kinds of wish in the query,
then counts how many each wine satisfies.

| Wish | How it's read | Example |
|---|---|---|
| Variety | Grape names from the catalogue, longest first | "sauvignon blanc" isn't also read as "sauvignon" |
| Country | Country names plus adjectives | "italian" → Italy |
| Colour | Everyday words | "pink" → rosé, "bubbly" → sparkling |
| Price | Explicit amounts only | "no more than $15" → at most 15 |

**Score** = number of wishes matched + points / 1000, the same trick as
Popularity: points only break ties.

- **Hand example from the tests:** for "french white under 20", a French
  white at $12 matches all three wishes and scores 3.085; the same wine at
  $25 matches two and scores 2.088.
- **A soft score, not a filter:** it always returns k wines, even when nothing
  matches everything.
- **No wish at all:** every wine scores 0 and the ranking falls back to points,
  i.e. "best-rated".

**"Why rules and not an LLM to read the query?"** CLAUDE.md rule 2: the LLM
never ranks. An LLM reading the query would make rankings depend on a
non-deterministic model and break reproducibility. Rules are crude but fully
explainable and always give the same answer.

**The colour table (B2):**

- **Why:** the dataset has no colour column, but colour is the most natural
  wish for a novice ("something red").
- **What:** a hand-made table maps the 71 most common varieties to a colour.
  Together they hold 92.2% of reviews (`top_share.csv`). Any variety not in
  the table has no colour.
- **Defence:** documented, limited to the common varieties, and checked by
  you or the sommelier. The uncertain entries are flagged in its `note`
  column (Moscato, Meritage, Port, Glera, Zinfandel).

**The details that show care:**

- **Accents removed before matching:** "gewurztraminer" finds Gewürztraminer.
- **Everyday words never read as names:** six variety names are skipped
  (Melon, Apple, Other, Mission, Debit, Vital), so "tastes like melon" isn't a
  request for the Melon grape. "us" is never read as the US.
- **Price wording:** "no more than 20" is read only as a maximum, not also as
  "more than 20".
- **No price, no match:** a wine without a price never satisfies a price wish,
  consistent with S0.

**Limitations to state yourself:**

- **Currency:** "under 20 euros" is compared with dollar prices, unconverted.
- **Vague words:** "cheap" is ignored, because any threshold would be
  arbitrary.
- **Colour words in other phrases:** "red fruit flavours" also reads as a wish
  for red.
- **Double counting:** "prosecco" counts twice for Prosecco wines, once as the
  variety and once as the colour sparkling.

## 10. `make eval` — the evaluation protocol

**What it does:**

1. Reads the casual queries: `wine_id` plus the query written from that
   wine's note.
2. Fits all four models with the test wines held out.
3. Asks each model for the top 5 for each query.
4. Computes every metric.
5. Writes `results/ablation.csv` (one row per model) and
   `results/ablation_by_band.csv` (per variety band).

**Why it refuses to run now:** the queries come in S2. Without them it stops
and says nothing was evaluated. No preliminary number can exist that might
later be "improved".

**The protocol decisions:**

- **Test wines are always held out,** even when evaluating on the dev set. The
  test set must never influence any model.
- **Per-band results:** this delivers the S0 decision, showing whether
  Semantic helps rare varieties as well as common ones.
- **Robustness (D):** tail share is also reported with the tail at ≤ 5 and
  ≤ 20 reviews. If Semantic beats Popularity at all three thresholds, the
  conclusion doesn't depend on the choice of 10.
- **ILD:** uses the SBERT embeddings for every model, the same yardstick for
  all. That's why `make eval` runs on the laptop, where the embeddings live.

**Novelty fix (C1, add-one smoothing):** p = (winery reviews + 1) / (reviews
counted + number of wineries).

- **Hand example from the tests:** wineries with 3, 1 and 0 counted reviews,
  out of 4 reviews and 3 wineries in total, get 4/7, 2/7 and 1/7. Every share
  is above 0, and they add up to 1.
- **Why not count on the full catalogue (C2)?** It would be inconsistent with
  the tail label, which deliberately ignores test wines. Smoothing keeps one
  rule everywhere.

## 11. Weaknesses to raise yourself (part 2)

1. **The Attribute reader is crude:** no currency conversion, no vague prices,
   and a colour table built by hand.
2. **The baselines aren't tuned.** That's deliberate, for fairness, but it
   means "untuned baseline vs untuned model", not "best possible baseline".
3. **Negation is hard for every model.** TF-IDF only matches the word "not";
   sentence models also handle "not too heavy" poorly. Don't claim Semantic
   solves negation. S2's refinement experiment ("less oaky") is where that is
   tested.
4. **TF-IDF's stop-word list was adjusted** (8 words kept). Say why before
   anyone asks: the plain list removed negations and "full", which would have
   handicapped the baseline.

## How the code is laid out (part 2)

| Piece | One line you should be able to say |
|---|---|
| `TfidfRecommender` | Word vectors weighted by rarity, learned without test wines; cosine ranking. |
| `keep_words` | Stop words that stay in, because they carry meaning ("not", "full", ...). |
| `normalise` | Lower case, no accents, letters and digits only, so names match however they're typed. |
| `find_names` | Finds catalogue names in the query, longest first, whole words only. |
| `parse_price` | Explicit amounts: under / over / between / "or less". |
| `AttributeRecommender.parse` | The wishes: varieties, countries, colours, price range. |
| `AttributeRecommender.score` | Wishes matched + points / 1000. |
| `variety_colour.csv` | Hand-made colour for the 71 most common varieties (to be checked). |
| `winery_counts` | Reviews per winery without test wines: the basis of Popularity, the tail and novelty. |
| `item_popularity` | Winery shares with add-one smoothing, for novelty. |
| `evaluate` | All models × all queries → every metric, overall and per band. |
| `make eval` / `make eval-dev` | One command → `results/ablation*.csv`; refuses without queries. |
