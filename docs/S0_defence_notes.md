# S0 — Defence notes

*Study notes for the oral defence, not thesis text. Every number below is
copied from a file in `results/eda/` (named next to it) as produced by
`make data` on 21 Oct 2026. If `make data` is re-run and a file changes, the
file is right and these notes are out of date (CLAUDE.md rule 3).*

---

## The 60-second version

> "Before building any recommender, I built a reproducible data pipeline. One
> command, `make data`, turns the raw Kaggle file into a clean catalogue,
> sets aside the evaluation wines, and produces every EDA figure and table.
> The EDA does two jobs: it shows there are no user–item interactions, which
> rules out collaborative filtering and justifies a content-based approach;
> and it measures how concentrated the catalogue is, so the long tail is
> defined from evidence before any model exists. Every choice is recorded
> with its reason in `DECISIONS.md`."

---

## 1. Reproducibility — "How do I know your numbers are real?"

- **One command:** `make data` runs `data.py` then `eda.py`. No notebook steps.
- **Fixed seed:** `seed: 42` in `config.yaml`, used for every random sample.
- **Input fingerprint:** the raw file's SHA-256 is stored in
  `data_manifest.json`, so a different input file is detectable.
- **Deterministic:** re-running gives byte-identical outputs.
- **Pinned versions** in `requirements.txt`.
- **No hand-typed numbers:** every figure cited comes from a CSV in `results/eda/`.

**"Where does your data come from?"** Kaggle, *Wine Reviews* by zackthoutt
(`winemag-data-130k-v2.csv`). The copy used was downloaded from the
TidyTuesday mirror because Kaggle was blocked in the cloud environment. It
was verified: 129,971 rows, and all 500 rows of the original sample appear in
it unchanged. The SHA-256 lets anyone confirm it equals the Kaggle file.
*(Do this comparison before the defence.)*

---

## 2. Cleaning — `data.py → clean()`

Each step logs rows before / removed / after to `cleaning_log.csv`.

| Step | Removed | Why |
|---|---|---|
| Empty description | 0 | The description is the main signal. No text, no content-based recommendation. |
| Exact duplicate description | 10,016 | See below. |
| Missing variety | 1 | Needed for the test-set stratification and the attribute baseline. |
| **Result** | **129,971 → 119,954** | |

**Why drop duplicates?** Two reasons (`duplicate_check.csv`):

1. **They are scraping artefacts.** 9,982 of the 10,016 (99.66%) have the same
   title, winery, points and price as the copy that is kept, so they are the
   same review scraped twice.
2. **They break the evaluation.** If a note appears twice, a query built from
   it has two equally correct answers, and HR@5 becomes ambiguous.

The other 34 duplicates are the same note on a different listing
(`duplicate_different_wines.csv`). Some are clearly the same wine with a
slightly different title ("Estate Syrah" vs "Estate Grown Syrah"), so 34 is
an upper bound on truly different wines. They still have to go, for the same
reason: no recommender could tell two wines with identical notes apart.

**Missing values: why not fill everything in?** (`missingness.csv`)

- **Price (6.99% missing):** left empty, with a `price_missing` flag. An
  invented price would mislead the attribute baseline and the "under €20"
  filter. A user asking for cheap wine shouldn't get one whose price was
  made up.
- **Text columns (country, region, taster, designation):** set to `"Unknown"`,
  with flags on region and taster. These are labels; there is nothing
  sensible to estimate.
- **`taster_twitter_handle`:** dropped, because it repeats the taster's name.
- **When missingness is measured:** after removing rows but before filling, so
  it describes the data the models actually use.

**`wine_id`** is the original row number of the Kaggle file. It is stable, so
the same wine has the same ID on every run and on every machine. That matters
because the test IDs are saved to disk.

---

## 3. The test and dev sets — `make_splits()`

**What:** 200 test wines and 100 dev wines, which never overlap.

- **Test** is used once, for the final numbers.
- **Dev** is for tuning settings (k, refinement step size).

Tuning on the test set would mean optimising for the exam you then report.

**Why test wines stay in the catalogue.** Self-retrieval takes a wine's
tasting note, rewrites it as a casual novice query, and checks whether the
model finds that wine. Remove the wine and there is nothing to find. So "held
out" means *nothing is fitted or tuned on these wines* (TF-IDF vocabulary,
popularity counts and so on, from S1 on). It does not mean they disappear
from the catalogue.

**Stratification by variety band.** Varieties are grouped by review count:
1–10, 11–100, 101–1,000, and more than 1,000.

- **Why:** a purely random sample would be almost all Pinot Noir and
  Chardonnay. The top 1% of varieties (8 grapes) hold 49.9% of reviews
  (`top_share.csv`). The test set would then say nothing about rare wines,
  which is exactly what the research question is about.
- **Allocation (option 1c):** each band gets at least 30 test wines (15 for
  dev), and the rest is shared in proportion to band size. Result:
  31 / 34 / 44 / 91 (`data_manifest.json`).
  - Pure proportional sampling would starve the rare bands.
  - Equal numbers per band would distort the overall average.
  - The minimum is a compromise that allows per-band results, reported
    alongside the overall result.
- **Rounding ("largest remainder"):** after the minimums, fractional shares
  are rounded down, and the leftover places go to the bands with the largest
  fractions. That guarantees the total is exactly 200.

**Why fixed bands, not tail-based bands?** The split was made before the tail
was defined, on purpose. If the test set depended on the tail definition, you
would be choosing the definition and the exam at the same time.

**"You stratified by variety, but your tail is defined by winery.
Inconsistent?"** No, the two do different jobs. Variety bands make sure rare
grape types are tested. The tail is a property measured afterwards. Check:
87 of the 200 test wines (43.5%) are tail wines, slightly more than the
tail's 36.4% share of the catalogue (`tail_summary.csv`), so the tail is not
under-represented.

**Why at least 20 words?** Only 1.68% of notes are shorter (`summary.csv`),
and a very short note gives any model too little to match against. Those
wines stay in the catalogue as possible recommendations; they just can't be
test questions.

**Why 200?** It comes from the scope document. Sanity check: the standard
error of a hit rate over 200 queries is at most √(0.25/200) ≈ 3.5 percentage
points. That is precise enough to compare models, but small differences
between models shouldn't be over-interpreted.

**"The band counts include the test wines. Isn't that leakage?"** No.
Counting reviews per variety to decide how to sample is part of the
experimental design, not something a model learns from. Leakage is a model
being fitted on test data, and that rule applies from S1 on.

---

## 4. The EDA — `eda.py`

**Long-tail plot** (`fig_rank_frequency.png`). Items are ranked from most to
least reviewed, with review counts on the y-axis. Both axes are logarithmic
because the range is huge.

- **Varieties:** the curve is steep. A few grapes have more than 10,000
  reviews, and most have a handful.
- **Wineries:** the curve is flatter. The biggest winery has only about 200
  reviews, but there are 16,755 wineries, and 4,582 of them have just one
  review (`summary.csv`).

**Lorenz curve and Gini** (`fig_lorenz.png`, `summary.csv`).

- **Lorenz curve:** sort items from fewest to most reviews, then plot the
  cumulative share of items against the cumulative share of reviews. The
  diagonal means perfect equality, and the further the curve sags below it,
  the more concentrated the catalogue.
- **Gini:** a number between 0 and 1 that summarises how far the curve sags.

  > G = 2·Σ(i·xᵢ) / (n·Σx) − (n+1)/n, with the counts xᵢ sorted ascending and i = 1…n.

- **Results:**
  - Varieties: **0.93**, extremely concentrated.
  - Wineries: **0.61**, concentrated but less so.
- **Hand check** (in `tests/test_metrics.py`, worth memorising): for counts
  [1, 2, 3], Σ(i·xᵢ) = 1 + 4 + 9 = 14, so G = 28/18 − 4/3 = 4/18 ≈ 0.22.
- **Edge cases:** equal counts give 0. One item holding everything gives the
  maximum, (n−1)/n.
- **Why `gini()` lives in `metrics.py`:** in S1 the same function measures how
  concentrated the *recommendations* are, so the catalogue and the models are
  judged with one yardstick.

**Top-share table** (`top_share.csv`). The plain-language version of the Gini:

| | Share of reviews |
|---|---|
| Top 1% of varieties | 49.9% |
| Top 10% of varieties | 92.2% |
| Top 20% of wineries | 66.1% |

**Taster coverage: the argument against collaborative filtering**
(`fig_taster_coverage.png`, `taster_coverage.csv`, `summary.csv`).

- **The numbers:**
  - 19 tasters, and 20.77% of reviews have no taster at all.
  - Each wine is reviewed exactly once.
  - Filled share of the taster × wine table: 0.042.
- **The argument:** collaborative filtering finds wines liked by people
  similar to you, and similarity comes from people rating the same items.
  - **No overlap:** with one review per wine, no two tasters ever rated the
    same wine, so similarity cannot be computed at all.
  - **Wrong users:** the tasters are 19 professional critics, not consumers.
  - **No history:** a new user has no ratings at all, the cold-start problem.

  So the system must work from content, the text of the tasting notes.

**Description length** (`fig_description_length.png`). The median note is
40 words. The histogram marks the 20-word cut-off.

**Scrape date** (`vintage_years.csv`, `summary.csv`). The latest vintage year
in the titles is 2017, a lower bound: the data cannot be older than its
newest wine. That fits Kaggle's v2 release. *Confirm the exact date on the
Kaggle page.* It also matters for the business chapter: the catalogue is old
and US-centric, which is why location and availability are Future Work.

---

## 5. The long-tail definition: a winery with ≤ 10 reviews

**Numbers** (`tail_summary.csv`):

- 13,706 wineries (81.8% of all wineries) are in the tail.
- They make 43,690 wines (36.4% of the catalogue).
- 87 of the 200 test wines are tail wines.

**Defence:**

1. **Simple:** one sentence that anyone can check.
2. **Grounded in the data:** a top-20% (Pareto) cut on wineries lands at 10
   reviews on this data, so the threshold isn't arbitrary.
3. **Usable:** about a third of the catalogue is in the tail, with enough test
   wines to report tail results separately.
4. **Consistent unit:** the Popularity baseline can rank by winery review
   count, so the baseline and the tail measure the same thing.

**Rejected alternatives** (see `DECISIONS.md`):

- **Head = wineries holding 20% or 50% of reviews:** 88–98% of wineries end
  up in the tail, too broad to say anything.
- **Variety as the unit:** varieties are so concentrated that the tail holds
  either very few wines or almost every variety.

**"Isn't 10 arbitrary?"** Any threshold is. This one was chosen on evidence,
and in S1 results can also be reported at 5 and 20 reviews, to show the
conclusions hold either way. That is a cheap robustness check worth adding.

---

## 6. Weaknesses to raise yourself, before the examiner does

1. **Near-duplicates aren't detected.** Only exact duplicate notes are
   removed; a note with one word changed stays in.
2. **The data comes from a mirror.** Closed by comparing the SHA-256 with a
   Kaggle download.
3. **The vintage regex is rough.** A brand name containing a year (e.g. "1865")
   could be miscounted. It is only used as evidence for the scrape date.
4. **The 30 / 15 minimums per band are a judgement call,** chosen to allow a
   rough per-band estimate. They are not statistically optimised.
5. **The catalogue is from 2017 and US-centric.** That limits real-world use
   (images, availability) but not the research question.

---

## How the code is laid out

| Function | One line you should be able to say |
|---|---|
| `load_raw` | Reads the CSV and keeps the original row number as a stable `wine_id`. |
| `clean` | Removes empty, duplicate and no-variety rows, logs each step, flags and fills missing values. |
| `duplicate_check` | Compares each removed duplicate with the kept copy: same wine scraped twice, or not? |
| `variety_band` | Labels each wine by how common its grape variety is. |
| `allocate` | Sets each band's quota: the minimum first, then the rest proportionally. |
| `stratified_sample` | Draws each band's quota at random with the fixed seed. |
| `make_splits` | Builds test first, then dev from what's left; only notes of 20+ words. |
| `tail_mask` | True if the wine's winery has at most 10 reviews. |
| `gini` | One number for how unequal a set of counts is. |
| `lorenz`, `top_share` | The curve and table behind the Gini. |
| `summary_table`, `tail_summary` | The headline numbers you cite, written to CSV. |
