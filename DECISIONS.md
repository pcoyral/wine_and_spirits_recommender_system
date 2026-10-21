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
