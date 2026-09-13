# SensoryRec — Technical Methodology

This document specifies the technical methodology in full: five recommender approaches are designed and compared, then combined into a recommended final architecture. The five approaches map directly onto the **ablation ladder** established in the project's academic scope (non-personalized → structured → lexical-text → semantic-text → hybrid), giving the written report a clean, examinerfriendly progression where each addition is independently justified and measured. All approaches operate on the Wine Enthusiast Reviews dataset (129,971 wines, 13 columns) and are evaluated under the self-retrieval protocol (held-out wines + paraphrased "queries") plus the sommelier panel, as previously scoped.

Throughout, **"query"** refers to a user's free-text taste description (e.g., *"something fruity but not too sweet, easy to drink with dinner, under $30"*), and **"wine vector"** refers to whatever representation a given approach uses for each of the 129,971 catalog items.

---

## 1. Popularity Recommender (Non-Personalized Baseline)

### Objective
Establish the **floor** against which all personalized approaches are measured. Tests the question: *is personalization — of any kind — adding value over simply recommending the best-reviewed wines?* This also represents the simplest possible "recommendation" a retailer could offer (a curated best-sellers/highest-rated shelf).

### Implementation
- Rank all 129,971 wines by `points` (descending). Optionally break ties by `price` (ascending), to avoid systematically recommending only the most expensive bottles.
- A slightly richer variant — **"popularity within category"** — computes the same ranking *within* each `price` tercile (low/medium/high) or each `variety`, giving the illusion of light personalization (a user picks a price band) without using any text or taste signal. This variant is included because it mirrors what an unguided novice user does today: pick a price range, take the top-rated bottle in it.
- No model training required; recomputed trivially whenever `points`/`price` data changes.

### Advantages
- Trivial to implement and fully reproducible; zero training time.
- No cold-start problem for users *or* items — works identically for a brand-new user on their first visit.
- A genuinely strong baseline in absolute terms: high-`points` wines are, on average, well-regarded across many palates, so this is not a "straw man."
- Provides the statistical reference point every other model must be compared against (paired significance tests, Section 7).

### Disadvantages
- **Zero personalization** — every user (or every user in a given price band) receives an identical list, regardless of stated preferences. Cannot accept natural-language input at all, so it does not engage with the core research question.
- Ignores the `description` field entirely — discards the richest part of the dataset.
- Reinforces popularity bias: the same small set of acclaimed wines (concentrated in a handful of regions/varieties) appears for everyone, which directly damages Coverage, Diversity, and Serendipity.
- Cannot support the refinement loop (RQ3) — there is nothing to refine.

### Evaluation Metrics
- **Precision@K / Recall@K / NDCG@K** under the self-retrieval protocol — expected to be at or near the floor, since whether a held-out wine is highly rated has little to do with whether it matches a specific (paraphrased) taste description.
- **Coverage** — expected to be very low (the same top-N wines are returned for most/all queries within a price band).
- **Diversity** — expected to be low (top-rated wines cluster by region/style, e.g., premium Bordeaux/Napa Cabernet).
- **Serendipity** — by construction, ≈0.
- *Primary role:* the row at the bottom of the ablation table that every other model must beat to justify its added complexity.

---

## 2. Content-Based Recommender (Structured Attributes)

### Objective
Represent **"the status quo"** — what a novice user must do today without SensoryRec: translate their preferences into wine vocabulary (variety, region, price range, minimum score) and filter/search on those terms. This model answers: *how good are recommendations when a user can and does specify structured wine attributes?*

### Implementation
- **Feature vector per wine** (per the Feature Engineering Strategy):
 - `variety`: one-hot encoded, restricted to the top ~30 varieties by frequency (covering the large majority of the 129,971 rows) plus an `"other"` bucket — full one-hot over all 707 varieties would be too sparse and would let rare varieties dominate distance calculations.
 - `country`: one-hot encoded (43 categories — manageable in full).
 - `region_1` / `province`: top-N one-hot + `"other"` bucket, or target-encoded by mean `points` (documented choice; target encoding risks leakage and must be computed only on the training partition).
 - `price`: log-transformed, then min-max scaled.
 - `points`: min-max scaled to [0,1].
- **User profile vector**: constructed either (a) directly from structured inputs the user selects (e.g., "Pinot Noir," "France," "$20–40," "≥90 points"), encoded in the same feature space, or (b) as the centroid of one or more "seed wines" the user names as liked.
- **Scoring**: cosine similarity between the user-profile vector and every wine's feature vector; rank descending. `price` can additionally be applied as a **hard constraint** (filter out wines above a stated ceiling) rather than a similarity dimension, since "I will not pay more than X" is typically a constraint, not a preference gradient.
- Library: scikit-learn (`OneHotEncoder`, `MinMaxScaler`, `cosine_similarity` or `NearestNeighbors`).

### Advantages
- **Transparent by construction** — every recommendation can be explained in terms of named, human-meaningful attributes ("recommended because it's a Pinot Noir from Oregon under $35, similar to your stated preferences").
- No item cold-start: every wine has these attributes from day one.
- Computationally trivial — feature vectors are low-dimensional (roughly 30–80 dimensions depending on encoding choices), so similarity search over 129,971 items is near-instant with plain NumPy.
- Directly operationalizes "Layer 1" of the original architecture and gives a concrete, citable example of a classic content-based recommender (S15/S16).

### Disadvantages
- **Requires the user to already possess wine vocabulary** to construct a profile — this model *is* the barrier SensoryRec exists to remove, not a solution to it. It is included specifically to quantify how much is lost by *not* having a natural-language interface.
- One-hot encodings treat categories as **equidistant**: "Pinot Noir" and "Gamay" (both light-bodied, high-acid reds) are exactly as "different" in this representation as "Pinot Noir" and "Cabernet Sauvignon" (a full-bodied, tannic red) — the representation cannot capture cross-variety stylistic similarity.
- Ignores `description` entirely — discards the sensory/stylistic information that distinguishes two wines of the *same* variety/region/price (which can vary enormously).
- High risk of **over-specialization / filter bubble**: cosine similarity over a handful of categorical+numeric dimensions tends to return near-identical wines (same variety, same region, similar price), which is exactly the "totally perfect content-based RS suggests nothing surprising" failure mode from Theme 1 of the literature review.

### Evaluation Metrics
- **Precision@K / Recall@K / NDCG@K** under self-retrieval: construct "queries" from a held-out wine's *own* structured attributes (variety, country, price band, points), minus the wine itself, and test whether the held-out wine (or a stylistically similar one) is retrieved.
- **Coverage** — moderate; bounded by how finely the categorical features partition the catalog (likely concentrated in dominant categories — Pinot Noir, Chardonnay, Cabernet Sauvignon, US/France/Italy — per the EDA).
- **Diversity** — likely low-to-moderate for the reason above (over-specialization).
- *Primary role:* the "status quo" comparator. Models 3–5 must demonstrably outperform this model **specifically on natural-language, non-technical queries** to justify SensoryRec's premise — this is the empirical heart of RQ2.

---

## 3. TF-IDF Recommender (Lexical Text Similarity)

### Objective
Test the **simplest possible** way to accept free-text input: classical lexical (bag-of-words/TF-IDF) matching between a user's query and each wine's `description`. This is the EDA report's own recommended next step, and it asks a sharp question: *is semantic embedding (Model 4) actually necessary, or does naive keyword matching already work?*

### Implementation
- Fit `sklearn.feature_extraction.text.TfidfVectorizer` on all 129,971 `description` strings: lowercase, English stopword removal, unigrams + bigrams (`ngram_range=(1,2)`, to capture phrases like "ripe tannins" or "crisp acidity"), `min_df=5` and `max_df=0.8` to drop both rare typos and near-universal terms (e.g., "wine," "flavors" — per the EDA word cloud, these dominate and add little discriminative value).
- This produces a sparse TF-IDF matrix of shape (129,971 × |vocabulary|), where |vocabulary| is typically in the 5,000–20,000 range after the `min_df`/`max_df` filters.
- At query time, transform the user's free-text query with the **same fitted vectorizer** (do not refit) to obtain a query vector in the identical space.
- **Scoring**: cosine similarity between the query vector and every row of the TF-IDF matrix; rank descending.
- Library: scikit-learn only (`TfidfVectorizer`, `cosine_similarity`) — no GPU, fast even at full scale.

### Advantages
- **Directly accepts free-text natural-language input** — this is the simplest version of "Model 2" and already represents a step-change over Models 1–2 in terms of addressing the core research question.
- Extremely cheap and fully interpretable: for any match, one can list the exact shared terms and their TF-IDF weights — useful both for debugging and as a sanity check for the explanation layer.
- Excellent baseline for isolating the *marginal value* of semantic embeddings: if TF-IDF performs nearly as well as SBERT, that is itself an important (if less exciting) finding; if SBERT clearly outperforms, it justifies the added complexity empirically rather than by assertion.
- No training beyond fitting the vectorizer (seconds, even on the full dataset).

### Disadvantages
- **The vocabulary-mismatch problem — TF-IDF's central, predictable failure mode, and precisely the problem SensoryRec exists to solve.** A casual query ("tastes like a fruit punch but dry") shares almost no vocabulary with professional tasting notes ("notes of black cherry and dried herbs, firm tannic structure"), so cosine similarity collapses toward zero even when the underlying wine is an excellent match. TF-IDF is therefore **expected to fail hardest exactly on the population SensoryRec targets** (novice users using casual language) while performing reasonably on "professional-style" queries that happen to reuse Wine-Enthusiast vocabulary.
- No notion of synonymy or semantic relatedness: "oaky," "vanilla," "toasty," and "buttery" are treated as unrelated dimensions, even though they are closely related sensory concepts.
- Sparse, high-dimensional vectors are brittle for short queries — a 10-word casual query may have only 1–2 non-zero entries after stopword removal, making similarity scores noisy and easily dominated by a single coincidental term match.
- Does not naturally support meaningful "iterative refinement via vector arithmetic" (RQ3): subtracting sparse TF-IDF vectors does not correspond to "less of a concept" the way dense semantic-embedding arithmetic can.

### Evaluation Metrics
- **Precision@K / Recall@K / NDCG@K** under self-retrieval — critically, **reported separately for two query subsets**: (a) "professional-style" paraphrases that retain some Wine-Enthusiast vocabulary, and (b) "casual" paraphrases that deliberately avoid it. This split is the **key diagnostic** for the entire methodology: TF-IDF is predicted to perform reasonably on (a) and poorly on (b), setting up the central contrast with SBERT (Model 4).
- **Coverage / Diversity / Serendipity** as before.
- **Additional diagnostic — query sparsity rate**: the proportion of casual queries that return *zero* non-zero-similarity wines (i.e., complete retrieval failure). This single number quantifies the vocabulary gap empirically and is one of the most quotable results in the whole evaluation chapter.

---

## 4. Sentence-BERT Recommender (Semantic Embedding)

### Objective
Directly target the vocabulary gap: embed both casual user queries and professional tasting notes into a **shared semantic space** (via SBERT `all-MiniLM-L6-v2`), where similarity reflects *meaning* rather than *exact word overlap*. This is the technical core of the project's novelty and the primary mechanism for RQ1–RQ3.

### Implementation
- **Offline (one-time batch job)**: encode all 129,971 `description` strings with `sentence-transformers`' `all-MiniLM-L6-v2` model → a (129,971 × 384) dense matrix of unit-normalized embeddings. At this scale (~200 MB as float32), the matrix fits comfortably in memory; exact cosine similarity via NumPy matrix multiplication is fast enough for the demo (no approximate-nearest-neighbor index such as FAISS is required at this scale, though it remains an option if the demo needs sub-100ms latency).
- **Online (per query)**: encode the user's free-text query with the same model → a single 384-d vector; compute cosine similarity (dot product, given unit-normalized vectors) against the full wine-embedding matrix; rank top-K.
- **Refinement mechanism (RQ3)**: for a follow-up phrase such as *"less oaky"*, derive a **direction vector** via the Word2Vec-style analogy principle (Mikolov et al., 2013 — to be added per the literature roadmap): e.g., `direction("oaky") = mean(embeddings of wines whose descriptions contain "oak"/"oaky"/"toast") − mean(embeddings of wines from the same variety/style cluster whose descriptions do not)`. The refined query becomes:

 `query_refined = normalize(query_embedding − α · direction("oaky"))`

 where `α` is a tunable refinement strength (e.g., 0.5–1.0, calibrated against the sommelier panel — does subtracting at α=1 actually produce wines the sommelier judges as "less oaky" relative to the original top-K?). Re-rank using `query_refined`.
- Library: `sentence-transformers` (`SentenceTransformer('all-MiniLM-L6-v2')`), NumPy for similarity computation.

### Advantages
- **Directly targets the vocabulary gap**: semantically similar phrases ("tastes like a fruit punch" / "jammy berry flavors, ripe and sweet") should land close together in embedding space even with **zero lexical overlap** — exactly the property TF-IDF lacks, and exactly the property the consumer-behaviour literature (Theme 7) says novice users need.
- Provides a **principled basis for the refinement mechanism** (RQ3): dense embedding spaces trained via contrastive/siamese objectives have been shown (in the broader embeddings literature) to support meaningful linear directions, which is the entire mechanism behind Model 5's "more/less" adjustments.
- Captures **cross-variety stylistic similarity** that Model 2 cannot: two wines of *different* varieties described in similar sensory language ("light-bodied, high acidity, red fruit, low tannin") will be embedded close together, even though their one-hot `variety` vectors are orthogonal.
- Pre-trained and ready to use — no fine-tuning required to obtain a working baseline within the capstone timeline; 384-d vectors are compact enough for fast, GPU-free inference at query time.

### Disadvantages
- **Reduced interpretability relative to TF-IDF**: there is no simple "shared term" explanation for *why* two embeddings are close — this is mitigated by the separate template-based explanation layer, but that explanation is post-hoc, not the actual mechanism, and this distinction should be stated explicitly in the report.
- **Domain-mismatch risk**: `all-MiniLM-L6-v2` is trained on general-purpose sentence-similarity corpora (web text, NLI, STS benchmarks), not wine-specific text. It may not cleanly separate fine-grained wine-specific concepts (e.g., "tannic" vs. "astringent," or "minerality" vs. "acidity") the way a domain-fine-tuned model would. This is a genuine limitation that should be reported honestly, not hidden.
- **Semantic drift risk**: Wine Enthusiast descriptions often mix sensory content with food-pairing suggestions, aging advice, and stylistic commentary ("drink now through 2026," "pair with grilled salmon"). SBERT embeds the *whole sentence*, so similarity may sometimes reflect topical overlap (e.g., shared food-pairing language) rather than pure sensory similarity — worth a qualitative spot-check during validation.
- **Refinement mechanism requires curated contrastive pairs** (e.g., "oaky" vs. "unoaked" within the same style cluster) — constructing these requires either keyword-based heuristic labeling of the corpus or manual curation, and the validity of the resulting direction vectors needs sommelier sign-off before being trusted.
- One-time embedding computation of 129,971 descriptions is a non-trivial batch job (minutes on GPU, longer on CPU) — manageable, but should be documented as a reproducibility step (e.g., run once, cache the resulting matrix to disk).

### Evaluation Metrics
- **Precision@K / Recall@K / NDCG@K** under self-retrieval, with the **same professional-vs-casual query split** used for TF-IDF — the core empirical test of RQ2 is the *difference in performance gap* between TF-IDF and SBERT on the casual-query subset specifically.
- **Coverage / Diversity / Serendipity.**
- **Sommelier panel**: qualitative validation that semantically-matched wines are genuinely stylistically appropriate (not merely topically related, per the semantic-drift concern above).
- **Embedding-space sanity check** (methodological, not a headline metric): a 2D projection (t-SNE/UMAP) of the wine-embedding matrix, colored by `variety` or by a coarse style label, to visually confirm that known wine-style groupings form coherent regions before relying on the space for refinement arithmetic.
- **Refinement-specific check** (RQ3, validated here before being folded into the hybrid): does `query_refined` actually shift the top-K toward sommelier-confirmed "less oaky" wines relative to `query_embedding`'s top-K?

---

## 5. Hybrid Recommender (Combined / Final Model)

### Objective
Combine the complementary strengths of structured-attribute similarity (Model 2 — precise on objective dimensions like price and region), semantic text similarity (Model 4 — bridges the vocabulary gap), and a quality prior (Model 1 — avoids recommending obscure-but-poorly-reviewed matches), into a single ranking function. This is the model the entire ablation ladder builds toward, and the one the refinement mechanism (RQ3) ultimately operates within.

### Implementation
- **Composite wine vector** (per the Feature Engineering Strategy's "Composite Taste Vector"): the SBERT description embedding (384-d) and the structured feature vector from Model 2 (≈30–80-d) are kept as **separate components** rather than naively concatenated, because they live on different scales and represent qualitatively different signals.
- **Scoring function** — a weighted sum of three normalized similarity components:

 `score(query, wine) = w1 · sim_SBERT(query, wine) + w2 · sim_structured(query, wine) + w3 · quality_prior(wine)`

 where:
 - `sim_SBERT` is the cosine similarity from Model 4 (already in [-1, 1], typically rescaled to [0, 1]);
 - `sim_structured` is the cosine similarity from Model 2 over whatever structured preferences the user *did* specify (if the user gives no structured input — the default, vocabulary-agnostic case — this term can be set to a neutral constant or simply dropped, i.e., `w2 = 0` for that query);
 - `quality_prior` is `points` (min-max scaled to [0,1]), acting as a tie-breaker/quality floor;
 - `w1, w2, w3` are weights, `w1 + w2 + w3 = 1`, with `w1` (SBERT) expected to dominate given the project's premise.
- **Hard constraints** (e.g., an explicit price ceiling mentioned in the query, extracted via a simple rule-based/regex parser — "under $30," "less than 30 dollars") are applied as a **pre-filter** before scoring, not as part of the weighted sum, since these are typically non-negotiable constraints rather than preference gradients.
- **Weight tuning**: `w1, w2, w3` are selected via grid search (or a simple learned re-ranker, e.g., logistic regression over the three component scores) against the **self-retrieval validation split**, with a strict train/validation/test separation to avoid overfitting the weights to the evaluation set that also produces the headline results.
- **Refinement loop (RQ3)**: operates on the `sim_SBERT` component exactly as in Model 4 (direction-vector adjustment of the query embedding), then the full hybrid `score()` is recomputed for re-ranking — i.e., refinement is a property of one component of the hybrid, not a separate model.
- **Optional fourth component — taster-proxy CF signal (RQ5)**: a small re-ranking adjustment derived from the `taster_name`/`points` matrix (per the Optimal Academic Scope document), added as `w4 · cf_signal(wine)` with a small, conservative `w4`, and explicitly flagged in the report as exploratory given the small number of "users" (≈15–20 professional tasters).
- **Explanation layer**: for any returned wine, the score decomposition `(w1·sim_SBERT, w2·sim_structured, w3·quality_prior)` is converted into a template explanation, e.g., *"Recommended because its tasting notes emphasize ripe red fruit and soft tannins, matching your description (primary reason); it also falls within your stated price range and carries a strong critical rating."*

### Advantages
- **Mitigates each component's individual weaknesses**: structured features add precision on objective attributes and avoid SBERT's domain-mismatch blind spots there; SBERT bridges the vocabulary gap that structured/TF-IDF approaches cannot; the quality prior prevents recommending obscure, poorly-reviewed wines purely because their description happens to match semantically.
- **Maps directly onto an established hybridization category** (feature-combination / weighted hybrid, per the literature review's Theme 1 taxonomy) — giving the methodology a precise theoretical label rather than an ad-hoc description.
- **Most likely to top the ablation table** on Precision/Recall/NDCG, while remaining tunable for Coverage/Diversity/Serendipity via the weights (e.g., periodically down-weighting `quality_prior` for a fraction of results to deliberately inject serendipitous picks — directly addresses RQ4).
- **The natural substrate for the explanation layer** (Section 6.3 of the academic scope) — the weighted decomposition *is* the explanation, which is a cleaner story than retrofitting explanations onto a black-box score.
- Since Models 1–4 are literally special cases of this scoring function (`w1=1,w2=w3=0` → Model 4; `w1=0,w2=1,w3=0` → Model 2; etc.), the **entire ablation table can be generated from one implementation** by varying weights — reducing code duplication and ensuring all comparisons are apples-to-apples.

### Disadvantages
- **Most complex component to implement and validate**: weight selection introduces a hyperparameter search that must be insulated from the final test set (risk of subtly overfitting `w1, w2, w3` to the same data used for the headline comparison if validation discipline slips).
- **Combining heterogeneous similarity scales** (dense 384-d cosine similarity vs. low-dimensional structured cosine similarity vs. a scalar quality prior) requires careful normalization — naive combination risks one signal dominating simply because its raw value range is larger.
- **Inherits, rather than eliminates, each component's limitations**: SBERT's domain-mismatch/semantic-drift issues and Model 2's over-specialization risk are both still present in the hybrid — the hybrid balances them but does not solve them, and the report should not overstate this.
- **Harder to diagnose**: if the hybrid underperforms a single component on some metric, attributing the cause (bad weights? bad normalization? conflicting signals? the optional CF term adding noise?) requires its own ablation-within-the-ablation.
- The optional taster-proxy CF term (RQ5) is small-n and exploratory; if weighted too aggressively it could inject noise disproportionate to its evidentiary value — must be weighted conservatively and reported with explicit caveats.

### Evaluation Metrics
- **Full suite as the headline result**: Precision@K / Recall@K / NDCG@K (primary), Coverage, Diversity, Serendipity — reported across the *same* professional-vs-casual query split used for Models 3–4, where the hybrid is expected to **close the casual-query gap** that TF-IDF showed and **match or exceed** SBERT alone on professional-style queries (via the structured/quality components).
- **Component ablation table**: since Models 1–4 are special cases of this scoring function, report each weight configuration's results in one table — this *is* the central results table of the thesis.
- **Sommelier panel**, extended to rate the **decomposed explanation** alongside the recommendation itself (does the stated breakdown match the sommelier's own reasoning about *why* the wine fits?).
- **Refinement-specific evaluation (RQ3)**: pre- vs. post-refinement Precision/NDCG on the full hybrid score (not just the SBERT component), since refinement must improve the *final* ranking users see.
- **Paired statistical tests** (Wilcoxon signed-rank) comparing the hybrid against each of Models 1–4 on a per-query basis, across all reported metrics — the hybrid serves as the reference model in every comparison.

---

## Comparative Summary

| | **1. Popularity** | **2. Content-Based (Structured)** | **3. TF-IDF** | **4. Sentence-BERT** | **5. Hybrid** |
|---|---|---|---|---|---|
| **Accepts free-text NL input?** | No | No | Yes | Yes | Yes |
| **Addresses vocabulary gap?** | N/A | No (requires wine vocabulary) | No (lexical only) | **Yes — core mechanism** | Yes (via SBERT component) |
| **User cold-start** | None | Requires structured profile | None (text only) | None (text only) | None |
| **Item cold-start** | None | None | None | None | None |
| **Explainability** | Trivial | High (named attributes) | High (shared terms) | Moderate (post-hoc template) | High (decomposed score) |
| **Computational cost** | Negligible | Low | Low (sparse, CPU) | Moderate (one-time embedding batch; fast inference) | Moderate (sum of 2–4) |
| **Expected performance — professional-style queries** | Low | Moderate | Moderate–High | High | **Highest** |
| **Expected performance — casual queries** | Low | Low | **Low (predicted failure mode)** | High | **Highest** |
| **Coverage** | Very low | Moderate | Moderate | Moderate–High | Tunable (via weights) |
| **Diversity** | Low | Low–Moderate | Moderate | Moderate–High | Tunable |
| **Serendipity** | ~0 | Low | Low–Moderate | Moderate | Tunable (deliberate via weights) |
| **Supports refinement (RQ3)?** | No | No | Weakly | **Yes (direction vectors)** | Yes (inherits from SBERT component) |
| **Role in thesis** | Floor baseline | "Status quo" comparator | Diagnostic for vocabulary gap | Core novel mechanism | **Final recommended architecture** |

---

## Recommended Final Architecture

**The Hybrid Recommender (Model 5) is the recommended final architecture for the SensoryRec prototype**, implemented such that Models 1–4 are generated as special cases of the same scoring function (Section 5's implementation notes) — meaning the "recommendation" of the hybrid is also a recommendation to build the comparison ladder as one parameterized system rather than five separate codebases.

### Architecture at a Glance

**Offline indexing pipeline (run once, cached):**
1. Load and clean the 129,971-row Wine Enthusiast dataset per the documented missingness strategy.
2. Compute structured feature vectors for every wine (one-hot top-30 varieties + "other," one-hot country, encoded region, log-price, scaled points) → Model 2's representation.
3. Compute SBERT (`all-MiniLM-L6-v2`) embeddings for every wine's `description` → Model 4's representation (384-d matrix, cached to disk).
4. Compute the quality prior (scaled `points`) → Model 1's signal.
5. (Optional) Construct the taster×wine matrix for the RQ5 exploratory CF signal.
6. Pre-fit the TF-IDF vectorizer on all descriptions → retained for the Model 3 comparison row in the ablation table (not part of the live serving path).

**Online query pipeline (per user request):**
1. Parse the free-text query for any explicit hard constraints (e.g., price ceiling) via simple rule-based extraction → applied as a pre-filter.
2. Encode the remaining free text with SBERT → query embedding.
3. (Optional) If the user also specified structured preferences, encode them in Model 2's feature space.
4. Compute `sim_SBERT`, `sim_structured` (if applicable), and retrieve `quality_prior` for all (filtered) wines.
5. Combine via the tuned weighted scoring function → rank top-K.
6. Generate the decomposed template explanation for each returned wine.

**Refinement loop:**
7. On follow-up feedback ("less oaky," "more like #2"), compute the relevant direction vector, adjust the query embedding (`α`-scaled subtraction/addition), and repeat steps 4–6.

### Why This Architecture Is Recommended

- It directly answers **RQ1** (the mechanism is the weighted hybrid combination, with SBERT as the vocabulary-bridging component — a concrete, implementable, citable design).
- It is structured so that **RQ2's comparison is built-in**, not bolted on afterward: Models 1–4 fall out of the same scoring function as parameter settings, so the ablation table is a natural product of the implementation rather than five separate efforts.
- **RQ3** (refinement) operates on a single, well-defined component (`sim_SBERT`) within a system whose other components remain stable — isolating the effect of refinement cleanly.
- **RQ4** (beyond-accuracy metrics) is directly tunable via the weights, giving the results chapter a natural "what happens if we deliberately trade some precision for diversity" experiment.
- **RQ5** (taster-proxy CF) slots in as an optional fourth additive term without requiring architectural changes — consistent with the "exploratory, clearly-flagged" framing recommended in the academic scope.
- The **explanation layer falls out of the architecture for free** — the weighted decomposition *is* the explanation, satisfying the Layer 5 ambition without a separate XAI subsystem.
- **Computationally realistic** for a part-time capstone: the only "heavy" step is the one-time SBERT embedding batch (129,971 descriptions); everything else (structured features, TF-IDF, weighted scoring, demo-time inference) runs comfortably on a laptop CPU.

### What Is Explicitly *Not* Part of the Final Architecture

- **TF-IDF (Model 3)** is retained purely as a **reported comparison point** in the ablation table — it is not part of the live serving path. Its purpose is to *demonstrate*, empirically, the marginal value of semantic embeddings over lexical matching (the casual-vs-professional query split), not to serve recommendations in the demo.
- **Full collaborative filtering on real user interaction data** remains future work (no such data exists); the taster-proxy CF term is included only as a small, conservatively-weighted, explicitly-flagged exploratory signal (RQ5), not as a core component.
- **Context-aware pre/post-filtering (Layer 4)** and **a full XAI framework beyond template decomposition (Layer 5 in its fullest form)** remain design-only future work, as previously scoped — the architecture above is deliberately built to be *extensible* in these directions (e.g., a context pre-filter would slot in before step 4 of the online pipeline) without requiring redesign.
