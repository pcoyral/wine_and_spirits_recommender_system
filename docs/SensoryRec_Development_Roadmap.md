# SensoryRec — Development Roadmap

This roadmap breaks the SensoryRec prototype into seven implementable milestones, each mapping directly onto a component of the technical methodology (Models 1–5, the evaluation framework, and the demo application). Complexity estimates assume a single part-time developer with working Python/scikit-learn fluency and basic familiarity with `sentence-transformers` and Streamlit; estimates are rough planning aids, not commitments.

## Repository Structure

All milestones populate a single repository with the following structure (established in Milestone 1, extended throughout):

```
sensoryrec/
├── data/
│   ├── raw/                     # winemag-data-130k-v2.csv
│   ├── processed/               # cleaned dataset, train/holdout splits
│   └── cache/                   # precomputed embeddings, TF-IDF matrix, structured features
├── src/
│   ├── data/                    # M1
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   ├── schema.py
│   │   └── split_data.py
│   ├── features/
│   │   ├── structured_features.py   # M2
│   │   ├── tfidf_features.py        # M3
│   │   └── sbert_features.py        # M4
│   ├── models/
│   │   ├── popularity.py            # M2 (baseline)
│   │   ├── content_based.py         # M2
│   │   ├── tfidf_recommender.py     # M3
│   │   ├── sbert_recommender.py     # M4
│   │   ├── refinement.py            # M4
│   │   ├── hybrid_recommender.py    # M5
│   │   ├── weight_tuning.py         # M5
│   │   └── taster_cf.py             # M5 (optional, RQ5)
│   ├── evaluation/                  # M6
│   │   ├── metrics.py
│   │   ├── self_retrieval.py
│   │   ├── ablation_runner.py
│   │   └── sommelier_panel.py
│   └── app/                         # M7
│       ├── streamlit_app.py
│       ├── ui_components.py
│       └── session_state.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_exploration.ipynb
│   ├── 03_weight_tuning.ipynb
│   └── 04_results_analysis.ipynb
├── data_dictionary/
│   ├── cleaning_log.md
│   ├── query_sets.json              # professional vs. casual paraphrase queries
│   ├── held_out_queries.json
│   ├── tuned_weights.json
│   └── sommelier_panel_template.xlsx
├── results/
│   ├── ablation_table.csv
│   ├── sommelier_panel_results.csv
│   └── statistical_tests.csv
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_refinement.py
│   ├── test_evaluation.py
│   └── test_app.py
├── requirements.txt
└── README.md
```

---

## Milestone 1 — Data Preprocessing

**Goal:** Transform the raw Wine Enthusiast CSV into a clean, documented, reproducible dataset with an index/holdout split suitable for the self-retrieval evaluation protocol.

### Expected Outputs
- `data/processed/wines_clean.parquet` — cleaned dataset, with documented missingness handling (drop rows missing `description`/`variety`; impute/flag missing `price`, `region_1`, `region_2`, `designation`, `taster_name`).
- `data/processed/index_set.parquet` and `data/processed/holdout_set.parquet` — a stratified ~250-wine holdout (by `variety`/`country`) reserved exclusively for self-retrieval evaluation; never used for fitting encoders/vectorizers/embeddings.
- `data_dictionary/cleaning_log.md` — documents every cleaning decision, feeding directly into the methodology chapter's reproducibility appendix.

### Code Modules
- `src/data/load_data.py` — `load_raw_wine_data(path) -> pd.DataFrame`
- `src/data/clean_data.py` — `clean_wine_data(df) -> pd.DataFrame`
- `src/data/schema.py` — a `Wine` dataclass/schema used for validation throughout the codebase
- `src/data/split_data.py` — `create_self_retrieval_split(df, n_holdout=250, seed=42) -> (index_df, holdout_df)`

### Estimated Complexity
**Low–Medium (~8–12 hours).** Most exploratory work is already done in the EDA Report — this milestone operationalizes known findings into reusable, tested code. The main effort is precisely documenting the missingness strategy so it's defensible in the methodology chapter.

### Dependencies
- Input: raw `winemag-data-130k-v2.csv`.
- Libraries: `pandas`, `pyarrow`.
- No dependency on other milestones — this is the foundation everything else builds on.

### Testing Strategy
- `test_load_data_shape()` — loaded raw data has the expected shape (129,971 × 13).
- `test_clean_data_no_nulls_in_required_fields()` — `description`, `variety`, `points`, and (post-imputation) `price` have no nulls.
- `test_clean_data_row_count_within_tolerance()` — cleaning doesn't silently drop more than a documented threshold of rows.
- `test_split_no_leakage()` — holdout wine IDs never appear in the index set.
- `test_split_stratification()` — holdout's `variety`/`country` distribution roughly matches the full dataset.

---

## Milestone 2 — Content-Based Recommender

**Goal:** Build the structured-attribute recommender (Model 2) representing the "status quo" — recommendations from a user-specified profile of variety, region, price, and rating.

### Expected Outputs
- `data/cache/structured_features.npy` — feature matrix for the index set (one-hot top-30 varieties + "other", one-hot country, encoded region, log-scaled price, scaled points).
- Working `ContentBasedRecommender` with a documented API.
- A notebook demonstrating example structured queries with manual sanity-checks against known wine styles.

### Code Modules
- `src/features/structured_features.py`
 - `fit_structured_encoder(index_df) -> StructuredEncoder` — fit **only on the index set**, never the holdout
 - `StructuredEncoder.transform(df) -> np.ndarray`
 - `StructuredEncoder.encode_profile(profile_dict) -> np.ndarray` — encodes a user-specified profile (e.g., `{"variety": "Pinot Noir", "country": "France", "price_max": 40, "min_points": 88}`)
- `src/models/content_based.py`
 - `class ContentBasedRecommender`: `recommend(profile_dict, k=5, price_ceiling=None) -> List[wine_id]`
- `src/models/popularity.py` (also delivered here as the Model 1 floor baseline)
 - `class PopularityRecommender`: `recommend(k=5, price_band=None) -> List[wine_id]`

### Estimated Complexity
**Low–Medium (~10–14 hours).** The trickiest part is correctly handling unseen categories at inference time (the "other" bucket for varieties outside the top 30, and region encoding for regions not seen during fitting). Similarity computation itself is trivial at ~80 dimensions.

### Dependencies
- Requires M1 (cleaned index/holdout split).
- Libraries: `scikit-learn` (`OneHotEncoder`, `MinMaxScaler`, `cosine_similarity`).

### Testing Strategy
- `test_structured_encoder_output_dimensions()` — feature matrix has the expected column count.
- `test_unseen_variety_maps_to_other()` — a variety outside the top 30 maps to "other" without error.
- `test_price_log_transform_monotonic()` — higher raw price → higher (or equal) transformed value.
- `test_content_based_returns_k_results()` — returns exactly `k` distinct wine IDs.
- `test_content_based_price_ceiling_respected()` — all returned wines satisfy the stated price ceiling.
- `test_content_based_deterministic()` — identical input → identical output.

---

## Milestone 3 — TF-IDF Recommender

**Goal:** Build the lexical-similarity baseline (Model 3), and curate the professional-vs-casual paraphrase query set that drives the central diagnostic for the whole project.

### Expected Outputs
- `data/cache/tfidf_vectorizer.pkl` and `data/cache/tfidf_matrix.npz` — fitted on index-set descriptions only.
- Working `TFIDFRecommender`, including the query-sparsity diagnostic.
- `data_dictionary/query_sets.json` — ~30–50 "professional-style" and ~30–50 "casual" paraphrase queries of held-out wine descriptions (shared with Milestone 6).

### Code Modules
- `src/features/tfidf_features.py`
 - `fit_tfidf_vectorizer(descriptions, ngram_range=(1,2), min_df=5, max_df=0.8, max_features=15000) -> TfidfVectorizer`
 - `transform_query(vectorizer, query_text) -> sparse_vector`
- `src/models/tfidf_recommender.py`
 - `class TFIDFRecommender`: `recommend(query_text, k=5, price_ceiling=None) -> List[wine_id]`
 - `query_sparsity(query_text) -> float` — fraction of zero-similarity wines for a given query, the core vocabulary-gap metric

### Estimated Complexity
**Low (~6–10 hours coding) + a substantial content-curation task.** The TF-IDF code itself is the simplest model in the project — scikit-learn handles everything. The real effort is writing ~60–100 realistic paraphrases (professional-style and casual) of held-out wine descriptions; this is a writing task that benefits from informal sommelier sanity-checking on the "casual" set's realism, and should be budgeted as calendar time, not just dev time.

### Dependencies
- Requires M1 (index-set descriptions, holdout descriptions to paraphrase).
- Libraries: `scikit-learn` (`TfidfVectorizer`).
- Soft dependency: early informal sommelier input on the realism of "casual" paraphrases.

### Testing Strategy
- `test_tfidf_query_transform_same_vocab()` — query vector dimensionality matches the fitted matrix.
- `test_tfidf_returns_k_results()`.
- `test_tfidf_sparsity_diagnostic_range()` — `query_sparsity()` ∈ [0, 1].
- `test_tfidf_casual_vs_professional_gap()` — on the curated query set, mean top-1 similarity is higher for professional-style queries than casual ones. **This test encodes the expected finding** — if it fails, that's a result worth discussing, not just a bug.

---

## Milestone 4 — Sentence-BERT Recommender

**Goal:** Build the semantic-embedding recommender (Model 4) — the project's core novel mechanism — including the direction-vector refinement capability (RQ3).

### Expected Outputs
- `data/cache/sbert_embeddings.npy` — (n_index × 384) unit-normalized matrix, cached to disk.
- `data_dictionary/direction_vectors.json` — precomputed refinement direction vectors (e.g., "oaky," "sweet," "tannic," "fruity") with metadata (number of contrastive examples, sommelier-validated: yes/no).
- Working `SBERTRecommender` and refinement module.
- A notebook with a t-SNE/UMAP visualization of the embedding space colored by `variety` (the embedding-space sanity check from the methodology).

### Code Modules
- `src/features/sbert_features.py`
 - `compute_embeddings(descriptions, model_name="all-MiniLM-L6-v2", batch_size=64) -> np.ndarray` — one-time batch job with progress logging and disk caching
 - `encode_query(model, query_text) -> np.ndarray`
- `src/models/sbert_recommender.py`
 - `class SBERTRecommender`: `recommend(query_text, k=5, query_embedding_override=None) -> List[wine_id]`
- `src/models/refinement.py`
 - `build_direction_vector(embedding_matrix, descriptions, concept_keyword, contrast_keyword=None, variety_filter=None) -> np.ndarray` — contrastive mean-difference (Mikolov-style)
 - `refine_query_embedding(query_embedding, direction_vector, alpha=0.7) -> np.ndarray`

### Estimated Complexity
**High (~20–28 hours).** Two distinct sources of complexity:
1. The one-time embedding batch job over ~129,700 descriptions is the first genuinely "heavy" compute step (budget ~30–60 minutes on CPU, faster on GPU/Colab — plus environment setup time).
2. The refinement/direction-vector mechanism is the highest-novelty, least-precedented code in the project. Expect iteration: initial keyword-based contrastive-pair selection will likely need refinement after inspecting results, and validating that direction vectors behave as intended requires a tight loop with qualitative inspection (ideally with the sommelier).

### Dependencies
- Requires M1 (index-set descriptions).
- Libraries: `sentence-transformers`, `torch` (CPU works; GPU/Colab recommended for the batch job).
- Conceptual dependency: the literature roadmap's Word2Vec/Mikolov addition should be read before building the refinement mechanism, since it's the citable justification for the approach.

### Testing Strategy
- `test_embedding_shape_and_normalization()` — matrix is (n × 384), each row unit-norm within tolerance.
- `test_embedding_cache_idempotent()` — recomputing from cache returns identical results.
- `test_sbert_returns_k_results()`.
- `test_sbert_casual_query_outperforms_tfidf()` — on the M3 query set, SBERT's top-1 similarity for casual queries exceeds TF-IDF's — **the headline RQ2 test**, runnable as soon as both models exist.
- `test_direction_vector_unit_norm()`.
- `test_refinement_changes_ranking()` — refined query produces a measurably different top-K than the unrefined query.
- `test_refinement_direction_validated_subset()` — for concepts with a small sommelier-labeled validation set, the direction vector separates labeled-positive from labeled-negative wines along its own axis.

---

## Milestone 5 — Hybrid Recommender

**Goal:** Combine M2 (structured) + M4 (SBERT) + a quality prior into the final weighted scoring function, with M1–M4 reproducible as special cases; optionally add the taster-proxy CF signal (RQ5).

### Expected Outputs
- Working `HybridRecommender` — the unified scoring engine.
- `data_dictionary/tuned_weights.json` — final weight configuration plus the validation metrics that justified it.
- A weight-tuning notebook — itself key evidence for the methodology chapter.
- (Optional) `src/models/taster_cf.py` + notebook for the RQ5 exploratory analysis.

### Code Modules
- `src/models/hybrid_recommender.py`
 - `class HybridRecommender`: `score(query, wine_id) -> float`; `recommend(query_text, structured_profile=None, k=5, weights_override=None) -> List[(wine_id, score, score_breakdown)]`
 - `normalize_component_scores(...)` — per-component normalization (min-max or z-score) before weighting — its own unit-tested function given how easily normalization bugs distort results
 - `generate_explanation(score_breakdown, wine_row) -> str` — template-based explanation (Layer 5)
- `src/models/weight_tuning.py`
 - `grid_search_weights(weight_grid, eval_fn, validation_queries) -> best_weights`
- `src/models/taster_cf.py` (optional)
 - `build_taster_item_matrix(df) -> matrix`; `compute_taster_cf_signal(matrix) -> per_wine_adjustment`

### Estimated Complexity
**Very High (~24–36 hours).** This is the integration milestone where bugs in M2–M4 surface. Normalizing across heterogeneous similarity scales (dense 384-d cosine vs. low-dimensional structured cosine vs. a scalar quality prior) is a known trap and needs careful, tested implementation. Weight tuning has a genuine methodological dependency on M6 (see Sequencing below) — the `HybridRecommender` *class* can be written and unit-tested earlier with placeholder weights, but cannot be *finalized* until M6's metrics exist.

### Dependencies
- Requires M2, M3 (for the ablation special-cases) and M4 (core component).
- **Circular dependency with M6**: resolved by building M6's pure metric functions early (in parallel with M2–M4), so they're available for M5's weight-tuning grid search; M6's full ablation table is the last piece completed, after M5's weights are tuned.

### Testing Strategy
- `test_hybrid_special_cases_match_components()` — `weights=(1,0,0)` reproduces M4's rankings exactly; `(0,1,0)` reproduces M2's; `(0,0,1)` reproduces a pure popularity ranking. **The single most important test in the codebase** — it validates the "ablation ladder = one parameterized system" design.
- `test_normalization_bounds()` — each normalized component score falls in the chosen range before weighting.
- `test_weights_sum_to_one()` — configuration validation.
- `test_hybrid_explanation_matches_dominant_component()` — generated explanation references the highest-weighted component.
- `test_full_pipeline_end_to_end()` — integration test: raw query string → ranked list with explanations, on a small synthetic sub-catalog (~50 wines) for speed.

---

## Milestone 6 — Evaluation Framework

**Goal:** Implement the self-retrieval protocol, all metrics, the ablation runner, and the sommelier-panel instruments — producing the thesis's headline results tables.

### Expected Outputs
- Tested implementations of Precision@K, Recall@K, NDCG@K, Coverage, Diversity, Serendipity.
- `src/evaluation/self_retrieval.py` — the held-out-wine + paraphrase-query harness.
- `data_dictionary/held_out_queries.json` — finalized query sets (started in M3) with held-out-wine ground-truth labels.
- `results/ablation_table.csv` — the central results table (Models 1–5 × all metrics × professional/casual splits).
- `data_dictionary/sommelier_panel_template.xlsx` — structured instrument for the expert validation session.
- `results/sommelier_panel_results.csv` and `results/statistical_tests.csv` (paired Wilcoxon, hybrid vs. each other model).

### Code Modules
- `src/evaluation/metrics.py` — `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `coverage`, `diversity`, `serendipity` — **pure functions with no model dependency**, buildable in parallel with M2–M4
- `src/evaluation/self_retrieval.py` — `evaluate_model(recommender, held_out_queries, k=5) -> metrics_dict`
- `src/evaluation/ablation_runner.py` — `run_ablation(models_dict, held_out_queries) -> pd.DataFrame`; `run_statistical_tests(results) -> pd.DataFrame`
- `src/evaluation/sommelier_panel.py` — `generate_panel_items(models_dict, queries_subset) -> panel_df`; `summarize_panel_results(completed_panel_df) -> summary`

### Estimated Complexity
**High (~18–26 hours of dev time, excluding the separate calendar time for sommelier sessions and query curation).** Individual metric formulas are well-known but easy to get subtly wrong (off-by-one in @K, log-base in NDCG, tie handling) — warrants unit testing against hand-computed examples. The self-retrieval harness's validity depends entirely on the quality of the held-out query/ground-truth construction, which is as much a methodological design task as a coding one.

### Dependencies
- `metrics.py` has **no model dependencies** — the earliest-startable piece of the evaluation framework, ideally begun alongside M2.
- `self_retrieval.py`/`ablation_runner.py` require at least M2–M4; the full ablation table requires M5.
- `sommelier_panel.py` requires working M4/M5 plus sommelier calendar availability (a non-technical dependency).

### Testing Strategy
- `test_precision_at_k_known_example()` / `test_ndcg_at_k_known_example()` — hand-computed reference values.
- `test_coverage_full_catalog()` — coverage = 1.0 when every wine is recommended at least once.
- `test_diversity_identical_items()` — diversity of identical embeddings = 0.
- `test_self_retrieval_perfect_model()` — a mock "oracle" recommender that always returns the held-out wine first achieves Precision@1 = 1.0, validating the harness itself before trusting it on real models.
- Integration test: `run_ablation()` on a small synthetic sub-catalog with known expected orderings, checking output shape and value ranges.

---

## Milestone 7 — Streamlit Application

**Goal:** Build the demo web application — the tangible artifact for the oral defense — exposing the hybrid recommender with refinement and explanations.

### Expected Outputs
- Runnable `src/app/streamlit_app.py` (`streamlit run src/app/streamlit_app.py`).
- A deployed instance (e.g., Streamlit Community Cloud) with a public URL for the defense.
- A short screen-recording/GIF of a sample interaction, for the written report's appendix and as a defense fallback.

### Code Modules
- `src/app/streamlit_app.py` — main page (free-text input, optional structured filters, results display, refinement input, "how this works" expander showing the score breakdown)
- `src/app/ui_components.py` — reusable rendering (wine card, score-breakdown chart)
- `src/app/session_state.py` — manages refinement state across Streamlit reruns (current/refined query embedding, refinement history)

### Estimated Complexity
**Medium (~12–18 hours).** Streamlit's basic layout is low-friction; the real complexity is **state management for the refinement loop** (Streamlit reruns the whole script on each interaction, so the current query embedding must persist in `st.session_state`). Loading the ~200MB embedding matrix at startup needs `@st.cache_resource` to avoid reloading on every interaction — get this right early to avoid a sluggish demo.

### Dependencies
- Requires M5 (hybrid recommender) for the live backend.
- **Can start earlier with a mocked backend**: UI/layout/session-state work can begin as soon as M1 is done, using a placeholder recommender returning random wines — the one milestone where early parallel work genuinely helps if the schedule is tight.
- Libraries: `streamlit`.

### Testing Strategy
- Smoke test via Streamlit's `AppTest` framework: app starts without error; a sample query returns 5 results; refinement input changes the results.
- Manual UX testing with 2–3 non-technical people — for this milestone, "does this feel usable to a novice?" is more informative than automated tests, and serves as a lightweight complement to (or substitute for, if time runs short) the formal 10–20 person user study.
- Performance check: time-to-first-result on cold vs. cached start, confirming `@st.cache_resource` is effective.

---

## Sequencing and Parallelization Notes

| Item | Earliest start | Why |
|---|---|---|
| M1 | Immediately | No dependencies |
| M2, M3, M4 | After M1 | Independent of each other — can be built in parallel if helpful |
| M6 metrics (`metrics.py`) | Alongside M2–M4 | Pure functions, no model dependency — start this early to avoid the M5↔M6 circular dependency later |
| M5 (code) | After M2–M4 exist | Needs all three components |
| M5 (weight tuning) | After M6 metrics + M5 code | This is the resolution point of the circular dependency |
| M6 (full ablation table, sommelier panel) | After M5 is tuned | Produces final headline results |
| M7 (UI scaffold) | After M1 (with mocked backend) | Genuinely parallelizable if time is tight |
| M7 (live backend) | After M5 | Final integration |

**Total estimated dev time: roughly 98–134 hours** across all seven milestones (excluding calendar-time-only items: sommelier sessions, query-paraphrase writing, user study). At 8–10 hours/week alongside full-time work, this is approximately **10–17 weeks of dev time** — comfortably fits within the Jan 25 (literature review) → Mar 20 (full draft) window if M1–M4 and M6-metrics are started in parallel during the literature-review period rather than after it.
