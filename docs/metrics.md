# What every metric column means

One page for the supervisor's question of 16 Nov 2026 ("what are those
columns?"). Code: `src/sensoryrec/metrics.py` and `src/sensoryrec/evaluate.py`;
every metric has a unit test with a hand-checked toy example
(`tests/test_metrics.py`). The definitions were fixed before any result
existed (DECISIONS.md, 21 Oct 2026). The numbers themselves are only in
`results/*.csv`.

Notation: each query gets a list of the top **k** wines (k = 5, the primary
cut-off, and k = 10). With Q queries there are **Q × k slots**.

## Accuracy (only where a query has one correct wine: the S2 test)

| Column | Meaning | Formula | Toy example |
|---|---|---|---|
| `hr@k` | Hit rate: share of queries whose correct wine is in the top k | (queries with a hit) / Q | 2 queries, target found in one top 5 → 0.5 |
| `ndcg@k` | Like HR, but a hit at rank 1 counts more than at rank 5 | mean over queries of 1 / log2(rank + 1), 0 if missed | target at rank 1 → 1.0; at rank 2 → 0.63; missed → 0 |

With a single correct wine per query, NDCG is HR with a discount for rank.
Neither says anything about the other, *similar* wines in the list: that is
why an LLM judge is added (step 2 of the plan).

## Breadth: how much of the catalogue is ever shown

| Column | Meaning | Formula | How to read |
|---|---|---|---|
| `slots`, `distinct_wines` | Recommendations made, and how many different wines they contain | Q × k; number of unique wine_ids | Popularity shows the same few wines to everyone |
| `coverage` | Share of the catalogue recommended at least once | distinct wines / 119,954 | **Ceiling = slots / 119,954**: 1,000 queries × 5 can reach at most ~4%. Compare models, not against 100%. |
| `winery_coverage` | Same for wineries | distinct wineries / 16,755 | |
| `variety_coverage` | Same for grape varieties | distinct varieties / 707 | The "28%" the supervisor saw |

## Long tail: whom the recommendations go to

| Column | Meaning | Formula |
|---|---|---|
| `tail_share` | Share of the **slots** filled by long-tail wines: wines whose winery has ≤ 10 reviews (the thesis's tail definition) | tail slots / all slots |
| `tail_share_le5`, `tail_share_le20` | The same with ≤ 5 and ≤ 20 reviews, to show the result does not depend on the threshold | |

**Coverage vs long-tail share** (the supervisor asked): coverage counts
*which* wines appear at least once; tail share counts *how often* the
recommendations go to small wineries. A model can show many different wines
that are all from famous wineries (high coverage, low tail share), or keep
returning a few small-winery wines (low coverage, high tail share). The tail
is 36.4% of the catalogue (`results/eda/tail_summary.csv`), the natural
reference for tail share.

## Concentration

| Column | Meaning | How to read |
|---|---|---|
| `gini_wine` | Gini coefficient of how often each wine is recommended, counting never-recommended wines as 0 | 0 = every wine recommended equally often; 1 = everything goes to one wine. With 5,000 slots over 120,000 wines it is close to 1 for every model, so it separates models little. |
| `gini_winery` | The same at winery level | More informative than at wine level |

The Gini is a measure of **inequality** (as for incomes), read from a Lorenz
curve. It is not the AUC of a classifier, even though both come from an area
on a plot.

## Novelty: how little-known the recommended wineries are

`novelty` = the average, over the wines in a list (then over lists), of
**−log2(p)**, where p is the share of all Wine Enthusiast reviews written about
that wine's winery (test wines not counted, plus one to every winery so that
p is never 0).

- A wine from the most-reviewed winery (211 reviews) has a large p → low novelty.
- A wine from a winery with a single review has a tiny p → high novelty.
- It is measured in bits: one bit more means the winery is half as often reviewed.

It does **not** mean "wines without reviews" (every wine has exactly one
review) and it does not use X-Wines.

## Diversity within a list

`ild` (intra-list diversity) = 1 − the average cosine similarity between
every pair of wines in a list, using their Sentence-BERT embeddings; averaged
over lists. 0 = the five wines mean the same thing; higher = more varied.

**Why it is lowest for the semantic models:** it is measured with
Sentence-BERT's own embeddings for every model (one yardstick for all). The
semantic models pick wines close to the query *in that space*, so they are
also close to each other: coherent lists by construction. A low ILD is not a
flaw in itself; a list of five reds for a red request should be similar.

## Coverage curve

`simulated_coverage_curve*.csv` and `fig_simulated_coverage*.png`: distinct
wines recommended after the first n queries. A flat curve = the model keeps
returning the same wines; a rising curve = more users would still discover
new wines.

## Which metrics apply where

| Study | Accuracy | Breadth, tail, Gini, novelty, ILD |
|---|---|---|
| S2 self-retrieval (200 test, 100 dev queries) | yes | yes |
| Simulated study (1,000 queries) | no correct wine, so no | yes |
| LLM judge (150 queries, step 2) | judged relevance, 1–5 | — |
