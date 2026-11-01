# Status on 1 Nov 2026: what exists and what doesn't

*Each number below comes from the file named next to it. There are no model
results yet (accuracy, long tail, refinement).*

## Against the plan for this meeting

| Planned for 2 Nov | Status |
|---|---|
| 1-page scope v2 | `1_scope_summary.md` |
| S0: full-dataset cleaning and EDA, long-tail stats, scrape date | Done. The latest vintage in the titles is 2017 (`results/eda/summary.csv`); the exact scrape date still needs confirming on Kaggle |
| S1 started: Popularity and SBERT on 3 to 5 example queries, metrics with unit tests | S1 is complete (all 4 recommenders, metrics, tests). Demo of all four models on 5 example queries: GitHub Actions run 41330968527 (Actions tab → Demo, summary page) |
| Storyboard of the worked example | `3_storyboard.md` |
| Ablation table "if ready" | Not yet. The pipeline exists, but the casual test queries (S2) don't |

## S0: data and EDA (PR #1)

- Cleaning (`results/eda/cleaning_log.csv`): 129,971 rows reduced to 119,954.
  I removed 10,016 exact duplicate notes, 99.66% of which are the same wine
  scraped twice (`duplicate_check.csv`), and 1 row with no variety.
- Long tail (`summary.csv`, `top_share.csv`): 16,755 wineries and 707
  varieties, with a Gini of 0.61 for wineries and 0.93 for varieties. The top 1%
  of varieties (8 grapes) hold 49.9% of reviews, and 4,582 wineries have a
  single review.
- Tail definition (`tail_summary.csv`, `tail_options.csv`): a wine is in the
  tail if its winery has 10 reviews or fewer. That covers 81.8% of wineries and
  36.4% of wines.
- Why no collaborative filtering (`summary.csv`, `taster_coverage.csv`): one
  review per wine, 19 tasters, and 20.8% of reviews with no taster.
- Hold-out (`results/eda/data_manifest.json`): 200 test wines and 100 dev
  wines, stratified by variety band so that rare grapes get tested.

## S1: models and metrics (PRs #2 to #5)

- Four models share one interface: Popularity (winery review count), TF-IDF,
  Attribute (rule-based on grape, country, colour and price), and Semantic
  (Sentence-BERT `all-MiniLM-L6-v2` with cosine similarity).
- Metrics: HR@5, NDCG@5, coverage, tail share, Gini (wine and winery level),
  novelty and intra-list diversity. Each one is tested against a hand-worked
  example.
- `make eval` builds the ablation table once the test queries exist.

## S2: evaluation queries and refinement (PRs #6, #7), code done but not run

- A local LLM (Llama 3.2 3B via Ollama, temperature 0) will write 300 casual
  queries, with every answer cached and committed. Each query is checked
  automatically for expert vocabulary (85 banned terms, approved one by one),
  the wine's name and the price. A query that fails is retried with feedback;
  if it still fails, it is flagged and excluded.
- Human validity check: about 30 novices write queries from memory for 30 test
  wines (`docs/human_query_protocol.md`). This is on hold until IE's ethics
  rules are clear.
- Refinement ("sweeter", "less oaky", "cheaper") uses direction vectors built
  from the training notes, with the step size chosen on dev. Success is scored
  with a separate word list and compared against random directions, along with
  the cost to the original wine's rank.

## How the work is done

- One command regenerates each result (`make data`, `make eval`, ...), with
  fixed seeds and pinned versions.
- `DECISIONS.md` dates and justifies every methodological choice.
- 7 pull requests so far, with each sprint code-reviewed independently, and 104
  unit tests.
- AI assistance: Claude Code wrote the code, and each commit says so. I made
  every methodological choice, and `docs/S0–S2_defence_notes.md` holds my notes
  on how each part works.

## Not done yet

- Casual test and dev queries (the LLM runs on my laptop), then the first
  ablation table.
- Human queries (after the ethics guidance) and the sommelier check.
- S3: the chat app with grounded explanations and the worked example.
