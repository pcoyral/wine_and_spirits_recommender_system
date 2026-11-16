# S2b — Defence notes (after the November 2026 supervisor meeting)

*Study notes for the oral defence, not thesis text. Every number comes from
a results file; this note names the file and the column instead of copying
values (CLAUDE.md rule 3). Decisions and rejected alternatives are in
`DECISIONS.md` (entries of 12, 14 and 16 Nov 2026).*

Files behind this note:

| What | File |
|---|---|
| The 1,000 queries, with the brief behind each | `data/queries/simulated_queries.csv` |
| Quality of the queries (retries, flags, length) | `results/simulated_queries_quality.csv` |
| Coverage and diversity per model | `results/simulated_coverage.csv` |
| Distinct wines after n queries | `results/simulated_coverage_curve.csv`, `results/fig_simulated_coverage.png` |
| Model versions used | `results/simulated_model_revisions.txt` |
| Every LLM answer | `data/llm_cache/` |

---

## The 60-second version

> "My supervisor asked whether the models really reach the whole catalogue,
> not only the famous wines. So I generated 1,000 realistic customer requests
> and measured, for each model, how many different wines, wineries and grape
> varieties it recommends, how much of that is long-tail wine, how unevenly
> recommendations are spread, and how varied each list is. These requests
> have no correct answer, so this study measures breadth and diversity, not
> accuracy; accuracy comes from the self-retrieval test of S2. To get 1,000
> *different* requests reproducibly, the variety is built first with a fixed
> seed, as combinations of wishes, and Gemini only turns each combination
> into a sentence. Every answer is cached and committed, and each one was
> checked for expert vocabulary. The study also adds a second, open embedding
> model next to Sentence-BERT, so the semantic approach is not judged on one
> model only."

---

## 1. Why simulated queries, and why they have no right answer

- The S2 test (200 queries, each written from one held-out wine) answers
  *"does the model find the wine the person described?"* It is small, and
  each query is tied to one wine.
- The supervisor's question is different: *across many people, does the
  system keep recommending the same few wines, or does it spread across the
  catalogue?* That needs many queries, but no "correct" wine for each.
- So the two tests complement each other: accuracy from S2, breadth and
  diversity from S2b. Never compare the two tables directly.

**Likely question:** *"Without a right answer, how do you know the
recommendations are good?"* → I don't, from this study alone. A model that
returned random wines would have excellent coverage. That is why coverage is
read **together with** the S2 accuracy results, and why Popularity (no
relevance at all, and the narrowest) is the reference point.

## 2. How the 1,000 queries were made

1. **Briefs, not free generation.** Asking an LLM at temperature 0 for
   "1,000 different requests" gives repeats. Instead `build_briefs` combines
   wishes from `data/simulation_facets.yaml` with the fixed seed: who is
   asking, the kind of wine, and optional sweetness, weight, flavours, feel,
   occasion, budget, country and one "also" wish. Duplicate briefs are
   redrawn, so all 1,000 are different.
2. **Plausible combinations.** After a trial of 20, some requests were
   impossible (a white wine tasting of coffee). Per kind of wine, the file now
   lists the wishes that are not offered (`not_for`); "any kind of wine" has
   no limits.
3. **The LLM only phrases.** Gemini (`gemini-3.5-flash-lite`, temperature 0,
   fixed seed) writes what the customer would say, in one or two sentences.
   It never sees a wine and never ranks anything (rule 2).
4. **Same checks as S2.** Expert words from `data/wine_vocab.txt` trigger a
   retry that names the words; a query still failing would be flagged and
   left out. Column `pct_flagged` in the quality file shows whether any were.
   No leak check here: there is no target wine whose name could leak.

**Known quirk:** every retry came from one wish of my own, the occasion "a
wedding toast", because "toast" is on the expert list. The retry removed it
each time (`first_expert` column in the queries file). Left as is (DECISIONS).

**Known limitation:** the queries are longer than a real novice's, because a
brief can carry up to six wishes (`words_mean`, `words_median` in the quality
file). A request with many constraints is easier for keyword models.

## 3. Why Gemini, and why that model

- The supervisor suggested Gemini (free tier, as in the course notebook).
- The notebook's model (`gemini-2.5-flash`) has been retired for new users.
- The free tier shown in Google AI Studio allows 20 requests a day for the
  Flash models and 500 a day for the Flash Lite models. 1,000 queries plus
  retries needed Flash Lite: three daily runs.
- The model name is part of the cache key, so every stored answer records
  which model wrote it.

## 4. What each metric says

All on the top 5 for each of the 1,000 queries (`slots` = 5,000
recommendations per model).

| Column | Plain meaning |
|---|---|
| `distinct_wines`, `coverage` | How many different wines appear at all (as a share of the catalogue) |
| `winery_coverage`, `variety_coverage` | The same for wineries and grape varieties |
| `tail_share` (and `_le5`, `_le20`) | Share of recommendations from small wineries (≤ 10 reviews; robustness at 5 and 20) |
| `gini_wine`, `gini_winery` | How unevenly recommendations are spread (1 = all on one item) |
| `novelty` | How unknown the recommended wineries are, in bits |
| `ild` | How different the wines within one list are from each other |

**Read coverage against its ceiling, not against 100%.** 5,000 slots cannot
cover ~120,000 wines: at most about 4% of the catalogue. Compare models with
each other.

**Gini at wine level is close to 1 for every model** for the same reason: most
wines can never be recommended by 5,000 slots. It barely separates the
models; `gini_winery` separates them better.

**The coverage curve** shows whether a model keeps finding new wines as more
people ask. A curve that is still rising at 1,000 queries means the study
has not saturated: more queries would still surface new wines.

## 5. What the results show (read the numbers from the files)

- **Popularity** recommends the same handful of wines to everyone: its curve
  is flat from the first query. This is the baseline the cold-start argument
  is built on.
- **Attribute** reaches far fewer wines than the text models. It only reads
  kind, country, grape and price from a query, so requests that differ in
  taste words get the same list.
- **TF-IDF and both embedding models** keep finding new wines. In this run
  the open model `semantic_bge` surfaces the most distinct wines, wineries
  and varieties, then Sentence-BERT (`semantic`), then TF-IDF.
- **Long tail:** every text model puts a share of its recommendations on
  small wineries; compare `tail_share` with the tail's share of the
  catalogue in `results/eda/tail_summary.csv`. Popularity puts none there.
- **ILD is lowest for the Sentence-BERT model, and that is partly built in.**
  ILD is measured with Sentence-BERT embeddings for every model (decision of
  21 Oct). The Sentence-BERT recommender picks wines that are close to the
  query *in that same space*, so they are also close to each other. A low
  ILD here means coherent lists, not necessarily poor variety. Say this
  before the examiner does.

## 6. The second embedding model

- `BAAI/bge-small-en-v1.5`: open, small, runs on GitHub's machine without a
  key. Same recommender code; only the model and a short instruction in
  front of the query (`query_prefix`, from the model card) change.
- It appears in every table, the demo and `make eval` (once the S2 queries
  exist), so the comparison is not limited to the simulated study.
- Gemini embeddings (`gemini-embedding-001`) are the next candidate. ~120,000
  notes is a lot for a free daily quota; whether batch requests (many notes
  per request) make it feasible is still to be checked.

## 7. Reproducibility

- `make simulate` rebuilds the same briefs with the seed and reads every
  answer from `data/llm_cache/`: no request is sent, the same queries come
  out.
- `make eval-simulated` recomputes every table and the figure.
- The model versions are pinned in `config.yaml` (16 Nov 2026), taken from
  the run that produced the results.
- The workflow "Simulated queries" runs both in the cloud and pushes the
  outputs to the branch `simulation-results`.

## 8. Questions to expect

- *"Did the LLM choose the wines?"* No. It wrote the customer's sentence
  from a brief; ranking is deterministic.
- *"Aren't 1,000 LLM queries biased?"* Yes, towards one model's phrasing and
  towards my wish lists. The wish lists are in the repo, and the human
  queries of S2 are the check on realism.
- *"Why is coverage so small?"* Because of the ceiling: 5,000 slots over
  ~120,000 wines.
- *"Why not more queries?"* The free quota; and the curve shows the ranking
  of the models is already stable well before 1,000.
- *"Isn't wider coverage just noise?"* Possibly; that is why coverage is
  only read together with accuracy (S2 self-retrieval).
