# Decisions

Methodological choices, with date and reason. Numbers that support a decision
come from files in `results/`, never typed by hand.

## 2026-10-21 — Data source
- `winemag-data-130k-v2.csv` (Wine Reviews by zackthoutt, Kaggle). Kaggle is
  blocked from the cloud environment, so this copy was downloaded from the
  R4DS TidyTuesday mirror (`rfordatascience/tidytuesday`,
  `data/2019/2019-05-28/winemag-data-130k-v2.csv`).
- Checks: 129,971 rows, the expected 14 columns, and all 500 rows of
  `data/raw/03_Sample_Data.csv` appear in it unchanged. The SHA-256 is in
  `results/eda/data_manifest.json`. **To do:** compare it with the SHA-256 of
  the file downloaded from Kaggle (`shasum -a 256 winemag-data-130k-v2.csv`).

## 2026-10-21 — Cleaning
- **Duplicates:** drop rows whose `description` is an exact duplicate (after
  trimming spaces), keeping the first. A repeated note is the same review
  scraped twice and would give two identical targets in self-retrieval.
- **Missing variety:** drop the row. Variety is needed for the test/dev
  stratification and for the attribute baseline.
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
  band as well as overall. *The minimums 30 / 15 were set by Claude as a
  starting value; change them in `config.yaml` if you prefer others.*
- **Eligibility:** only notes with at least 20 words can be picked for test or
  dev; very short notes give the retriever too little to match. They stay in
  the catalogue.
- **Test/dev wines stay in the catalogue** as retrieval targets. Following
  CLAUDE.md rule 4, anything *fitted* in S1+ (TF-IDF vocabulary, popularity
  counts, attribute stats, direction vectors) is fitted on the catalogue
  minus test wines.

## Open
- **Tail definition** — to choose after `make data` produces the EDA numbers.
  Options: (A) share-based Pareto cut on wineries, (B) fixed threshold
  (winery with ≤ k reviews), (C) head / mid / tail by cumulative review share.
  Must use the same unit (winery or variety) as the Popularity baseline.
- **Scrape date** — check the Kaggle dataset page; `latest_vintage_in_titles`
  in `results/eda/summary.csv` is a lower bound.
