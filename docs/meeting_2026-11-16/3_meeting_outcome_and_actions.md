# Meeting 16 Nov 2026: outcome and actions

From the transcript of the call with Santiago Gil Begue and the thesis he
pointed to (Zhen, *Multimodal conversational recommender for Zara*, section
3.6 and chapter 4). Next meeting: in about two weeks (~30 Nov 2026).

## What the supervisor said

- Positive overall: everything asked last time was done; "you already have an
  MVP to show". He liked the prototype (prompt bar, categories, map of Madrid).
- Roger Voss: explained, "makes sense".
- X-Wines: "this one we'll get more out of"; the long tail "gives you a lot of
  argument". He sees two datasets of the same shape (user, wine, rating):
  19 critics vs. a million consumers, and suggested combining them.
- Simulated study: semantic models beating TF-IDF and Attribute on breadth
  "is great". But 28% of varieties reached is still low for a thesis about
  discovery: **try top 10 instead of top 5**.
- The queries look varied enough; near-duplicates are fine.

## The four options for the next two weeks (he said: choose)

1. **Collaborative filtering**: X-Wines has users, so add a CF model to the
   comparison.
2. **Explain the pending metric columns**: long-tail share, novelty, diversity
   in a list; what each is and how it differs from coverage.
3. **LLM-as-a-judge**: a second LLM rates how good the 5 recommendations are
   for each query. One judge is enough to start.
4. **Images**: bottle images; he asked what information an image adds (your
   answer: the bottle shape tells Bordeaux from Burgundy, useful for learning).

Also mentioned, not assigned:
- With user feedback (X-Wines ratings) you could compute NDCG and similar.
- Scaling from 1,000 to 10,000 simulated queries (the free Gemini limit is the
  constraint).
- Zhen generated basic / medium / complex queries instead of one kind.

## Things said in the meeting that need correcting next time

| Said | Correct (source) |
|---|---|
| France and Portugal are ~70% of the dataset | 21.3%: the US is 42.1%, France 17.0% (`wines.parquet`) |
| About 10 sommeliers | 19 named critics; 20.8% of reviews unnamed (`results/eda/summary.csv`) |
| "15% of wineries covered, 85% missing" | Best winery coverage on the simulated study is 9.2% (bge-small); the ceiling with 5,000 slots is ~4% of wines (`results/simulated_coverage.csv`) |
| Novelty = wines that have no reviews / comes from X-Wines | Novelty = how little-known the recommended wineries are: −log2 of the winery's share of reviews in Wine Enthusiast, averaged over the list. Higher = less popular wineries. Nothing from X-Wines. |
| Long-tail share is like coverage | Different: coverage = share of the catalogue that appears at least once; long-tail share = share of recommendation slots filled by wines from small wineries (≤ 10 reviews) |
| Gini read as "like an AUC" | Gini here measures inequality of ratings across wineries (0 = equal, 1 = all on one); not a classifier AUC |
| "I added Gemini" | Gemini wrote the 1,000 simulated queries; Gemini *embeddings* are not done yet |
| "Combine the two datasets" | No shared IDs: Wine Enthusiast and X-Wines wines would have to be matched by winery, name and vintage, only partly possible |

## Scope points to settle (CLAUDE.md, needs the student's decision)

- **Collaborative filtering was OUT of scope.** The supervisor now suggests it.
  If chosen, record the scope change in `DECISIONS.md` with today's date. It
  can only run on X-Wines (Wine Enthusiast has one review per wine), so it is
  a second experiment, not a fifth model in the same table, unless the wines
  are matched.
- **Images:** no scraping of sites whose terms forbid it. X-Wines publishes
  label images (CC0) for its 1,007-wine Slim version and its 100-wine Test
  version (`XWines_Slim_1K_labels.zip` on the authors' Drive); a curated
  subset from it would satisfy the rule. Every image's source and licence go in `data/images/sources.csv`.
- **LLM-as-a-judge** fits the rules (the LLM evaluates, it does not rank) if
  the judge is cached, temperature 0, model pinned.

## How Zhen did LLM-as-a-judge (to copy, scaled down)

- One judge model, single-LLM setup; prompts combine chain-of-thought,
  a 1–5 Likert scale and an explicit rubric per criterion; scores shown on
  1–10 for readability.
- Levels and criteria: parsed query (completeness, no hallucinated
  constraints, occasion detected); each item (relevance to the query,
  occasion appropriateness, explanation quality); the whole set (answer
  relevance); pairwise model comparison (win / tie rates).
- 150 synthetic queries, 50 per tier: basic (1–2 constraints), medium (3–4,
  noisier), complex (an occasion + 2 constraints, conversational).
- A random baseline and an ablation give the scores meaning.

**For SensoryRec:** our 1,000 briefs already record their wishes, so they can
be split into basic / medium / complex tiers without generating new queries.
A judge sees the request and the five tasting notes and scores relevance per
wine and for the set. With the free Gemini quota (500 requests a day), 150
queries × 5 models at set level is 750 calls, about 2 days. A sommelier check
on a small sample would validate the judge.

## Recommended plan for two weeks (to confirm)

1. **Quick, low risk (1 day):** top 10 next to top 5 in the simulated study;
   one page defining every metric column.
2. **Main item: LLM-as-a-judge** on 150 stratified queries for all models,
   set-level relevance first. It answers the accuracy problem (the exact-wine
   test penalises good but different wines) and follows a thesis he rated
   highly.
3. **If time remains:** collaborative filtering on X-Wines as a separate
   experiment (scope change to record), or Gemini embeddings.
4. **Later:** images (increment A) from the X-Wines label images.
