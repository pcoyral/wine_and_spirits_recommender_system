# Decisions

Methodological choices, with date and reason. Numbers that support a decision
come from files in `results/`, never typed by hand.

## 2026-10-21 — Data source
- `winemag-data-130k-v2.csv` (Wine Reviews by zackthoutt, Kaggle). Kaggle is
  blocked from the cloud environment, so this copy was downloaded from the
  R4DS TidyTuesday mirror (`rfordatascience/tidytuesday`,
  `data/2019/2019-05-28/winemag-data-130k-v2.csv`).
- Checks: the row count (`raw_rows` in `results/eda/data_manifest.json`)
  matches the ~129,971 in CLAUDE.md; the 14 expected columns are present; and
  in a one-off check on 21 Oct 2026, every row of
  `data/raw/03_Sample_Data.csv` was found in it unchanged. The SHA-256 is in
  `results/eda/data_manifest.json`. **To do:** compare it with the SHA-256 of
  the file downloaded from Kaggle (`shasum -a 256 winemag-data-130k-v2.csv`).

## 2026-10-21 — Cleaning
- **Duplicates:** drop rows whose `description` is an exact duplicate (after
  trimming spaces), keeping the first. A repeated note is the same review
  scraped twice and would give two identical targets in self-retrieval.
  `results/eda/duplicate_check.csv` compares each removed duplicate with the
  kept copy (same title, winery, points and price = same wine).
- **Missing variety:** drop the row. Variety is needed for the test/dev
  stratification and for the attribute baseline. This step runs *before*
  de-duplication, so a duplicate that has a variety is never dropped in
  favour of a copy without one (code review, 21 Oct 2026).
- **Missing price:** keep the row, do not impute; add `price_missing`. An
  invented price would distort the attribute baseline and price filters.
- **Missing country, province, region, designation, taster:** fill with
  `"Unknown"`; flags `region_1_missing`, `taster_name_missing`.
- **`taster_twitter_handle`:** dropped (duplicates `taster_name`).
- **`wine_id`:** the original row index of the Kaggle CSV, so IDs are stable
  across runs. The raw file's SHA-256 is saved in
  `results/eda/data_manifest.json` to detect a different file.

## 2026-10-21 — Test and dev hold-out
- **Test set:** 200 wines. **Dev set:** 100 wines, disjoint from test, for
  tuning settings (k, refinement step size, ...). The test set is used only
  for the final numbers.
- **Stratification:** by variety frequency band, using the number of reviews
  of the wine's variety: 1–10, 11–100, 101–1,000, >1,000. Fixed bands, so the
  split does not depend on the tail definition chosen later.
- **Allocation (option 1c):** every band gets a minimum (test 30, dev 15),
  the rest is shared in proportion to band size. Results will be reported per
  band as well as overall. The minimums were proposed by Claude and
  **confirmed by the student on 21 Oct 2026**: 30 per band is enough for a
  rough per-band estimate while leaving most of the 200 proportional.
  `allocate()` raises an error if the settings cannot be met, so the sizes are
  always exactly 200 / 100.
- **Eligibility:** only notes with at least 20 words can be picked for test or
  dev; very short notes give the retriever too little to match. They stay in
  the catalogue.
- **Test/dev wines stay in the catalogue** as retrieval targets. Following
  CLAUDE.md rule 4, anything *fitted* in S1+ (TF-IDF vocabulary, popularity
  counts, attribute stats, direction vectors) is fitted on the catalogue
  minus test wines.

## 2026-10-21 — Long-tail definition
- **Decision:** a wine is in the long tail if its **winery has at most 10
  reviews** (`tail` in `config.yaml`). Size of the tail:
  `results/eda/tail_summary.csv`.
- **Options considered:** all in `results/eda/tail_options.csv` (written by
  `make data`), for both wineries and varieties:
  - (A) Pareto cut: items outside the top 20% by review count. For wineries
    the head starts at 10 reviews, so the A tail (`tail_max_reviews` 9)
    nearly coincides with the choice.
  - (B) Fixed threshold, winery with ≤ k reviews (k = 5, 10, 20). **Chosen
    with k = 10.**
  - (C) Head = most-reviewed items holding the first 20% / 50% of reviews.
    For wineries this puts most wineries in the tail (`tail_items_pct`), too
    broad to be informative.
  - Variety as the unit was rejected: varieties are so concentrated that the
    variety tail holds either very few wines or almost all varieties
    (`tail_wines_pct`, `tail_items_pct`).
- **Why:** one sentence to state and defend; close to the 80/20 split; about
  a third of wines and enough test wines fall in the tail to report results
  for the tail separately; the winery is also the unit the Popularity
  baseline can count, so baseline and tail are measured the same way.
- **From S1 on:** the winery counts behind the tail label are taken from the
  catalogue minus test wines (CLAUDE.md rule 4). The S0 EDA uses the full
  catalogue, which is descriptive only.

## Open
- **Scrape date** — check the Kaggle dataset page; `latest_vintage_in_titles`
  in `results/eda/summary.csv` is a lower bound. The vintage is the first year
  after the winery name in the title, so years in winery or cuvée names
  ("1912 Winemakers", "Vignes de 1939") are not counted.

## 2026-10-21 — S1: recommenders
- **Interface** (`recommenders/base.py`): `fit(catalogue, holdout_ids)`,
  `score(query)`, `recommend(query, k, filters)`. `recommend()` is shared, so
  models differ only in how they score. Ties are broken by `wine_id`.
- **Filters:** `max_price`, `min_price`, `country`, `variety`. A wine without a
  price never passes a price filter (its price is unknown, not cheap).
- **Popularity (option 1a):** score = number of reviews of the wine's winery,
  counted on the catalogue **minus test wines**; points break ties.
  Alternatives: (b) points only — "most acclaimed", unrelated to the tail
  definition; (c) like (a) but one wine per winery — closer to a shop's
  bestseller list, but adds a diversity rule to the baseline and distorts the
  ILD comparison. Why (a): same unit as the long-tail definition, and it is
  what a popularity baseline does — the same list for every query.
- **Semantic:** `sentence-transformers/all-MiniLM-L6-v2`, embeddings scaled to
  length 1 so cosine similarity is a dot product. The model is pretrained and
  learns nothing from our data, so test wines are embedded too (as retrieval
  targets). Embeddings are cached with a fingerprint of model, wine_ids and
  notes. **To do:** pin the model `revision` in `config.yaml` after the first
  download.
- **Where the model runs:** `make embed` / `make demo` run on the student's
  laptop; the cloud environment cannot reach huggingface.co.

## 2026-10-21 — S1: metric definitions
Fixed before any results exist, so the choice cannot be tuned to the results.
- **HR@k, NDCG@k:** one relevant wine per query (self-retrieval target).
- **Catalogue coverage:** distinct recommended wines / catalogue size.
- **Tail share:** share of recommendation slots filled by tail wines.
- **Gini of recommendations:** reported at **both wine and winery level**.
  With 200 queries × 5 slots over ~120k wines, almost every wine is never
  recommended, so the wine-level Gini is close to 1 for every model; the
  winery level makes differences easier to see.
- **Novelty:** mean of −log2(p), with p = the wine's winery share of all
  reviews. Each wine has exactly one review, so wine-level popularity would
  give every wine the same novelty.
- **Intra-list diversity:** 1 − mean pairwise cosine of the k wines, using the
  SBERT embeddings **for every model**, so all are measured with one
  yardstick. Limitation: the yardstick is the Semantic model's own space.
- **Open for S2:** p for novelty and the tail label are counted on the
  catalogue minus test wines, so a test wine whose winery has no other
  reviews gets p = 0 (−log2 0 is infinite). Decide how to handle it
  (e.g. add-one smoothing) before computing any numbers.
