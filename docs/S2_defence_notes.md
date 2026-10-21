# S2 — Defence notes

*Study notes for the oral defence, not thesis text. At the time of writing,
no S2 results exist yet: the queries, `make eval`, `make refine` and
`make compare-human` run on the student's laptop. Quote numbers only from
`results/*.csv` (CLAUDE.md rule 3). Decisions and their alternatives are in
`DECISIONS.md` (S2 entries, 21 Oct 2026).*

---

## The 60-second version

> "S2 builds the test that every model is judged on, plus the refinement
> feature. A local LLM rewrote each held-out tasting note as a short
> description in a novice's words. Every query is checked automatically: no
> expert vocabulary, no name of the wine, no prices. If a query fails, the
> LLM is told which words were not allowed and tries again. About 30 queries
> written by real novices, under a protocol where the note is hidden before
> they write, check whether the LLM queries are realistic. For refinement,
> 'sweeter' or 'less oaky' moves the query along a direction in embedding
> space learned from training notes. Whether it worked is judged with a
> *separate* word list, against two controls: random directions, and how far
> the original wine drops. The step size is chosen on dev data only. Every
> decision is in DECISIONS.md, and every number will come from a results file."

---

## 1. Why an LLM writes the test queries

**The problem:** self-retrieval needs, for each test wine, a query a novice
might write about it. 200 test queries plus 100 dev queries is too many to
collect from people within a capstone.

**Why a local model** (Llama 3.2 3B via Ollama, D2):

- **Reproducible:** the exact model version is pinned, and temperature 0 with
  a fixed seed gives the same answer on the same machine.
- **Free and private:** no API cost, and the data never leaves the laptop.
- **The cache is the record:** reproducibility isn't guaranteed across
  machines or Ollama versions, so every answer is stored in
  `data/llm_cache/` and committed (rule 2).

**"Doesn't an LLM in the evaluation make it unreliable?"** The LLM only
*writes the test questions*. It never ranks wines (rule 2). Rankings come
from deterministic scores, and the questions are frozen in the cache, so
anyone re-running `make eval` gets identical numbers.

---

## 2. The biggest risk — circularity, and how it's controlled

The LLM writes while looking at the tasting note. If it copies the critic's
words, keyword and embedding models find the wine easily, and accuracy looks
better than it would with real novices. That is the "circularity" risk in
the scope document. Four defences:

1. **Vocabulary check (D3):** the 85 expert terms in the `[expert]` section
   of `data/wine_vocab.txt` are banned (tannin, palate, finish, oaky, stone
   fruit, ...). The student approved every word; borderline words such as
   "complex" and "elegant" are deliberately banned too.
2. **Leak check:** the query may not contain its own wine's grape, winery,
   region, province, country or designation. Otherwise "a Pinot Noir from
   Sancerre" would hand the Attribute and TF-IDF models a shortcut.
3. **Copy share:** the share of a query's content words that also appear in
   its note. This *measures* copying, for LLM and human queries alike,
   instead of assuming it away.
4. **Human subset:** about 30 queries from real novices on the same wines. If
   the models find the wines far more easily from LLM queries than from
   human ones, the LLM queries were too easy, and the thesis must say so.

**The checks in detail:**

- **Matching:** whole words, ignoring case and accents, with plurals counted
  ("tannins" counts as "tannin").
- **Numbers:** any digit or currency symbol counts as a price slipping in,
  which D7 forbids.
- **Colour words are allowed** (D7). "Red", "white" and "sparkling" are
  everyday words. Banning them would leave the Attribute model almost
  nothing to read, and an examiner could say it was set up to fail.

**What happens when a query fails:**

- The retry prompt quotes the previous answer, names the words that weren't
  allowed, and asks for a rewrite, up to 3 times.
- **Why not simply ask again?** With temperature 0, the same prompt gives the
  same answer. The feedback changes the prompt deterministically, and it is
  cached too.
- **Early stop:** if the model repeats the same answer, it stops trying,
  because the next retry prompt would be identical.
- **Flagged queries:** a query that still fails is flagged, *left out* of the
  evaluation (rule 5), and the flag rate is reported in
  `results/queries_quality.csv`. Hiding failures would be worse than
  reporting them.

**"Why 'taste only' (D7)?"** Each query must describe the target wine, and
food, occasion or price don't identify a wine from its tasting note. Those
wishes belong to the chat and refinement stage, not to the retrieval test.

---

## 3. The human queries (D6)

**The protocol** (`docs/human_query_protocol.md`):

1. The participant reads the note once, for at most 60 seconds.
2. The card is taken away.
3. They describe the taste in their own words.

**Why hide the note?** Writing from memory forces them into their own
vocabulary instead of copying. That is exactly what we want to compare with
the LLM, which *does* see the note: an honest asymmetry to mention.

**The details that show care:**

- **Who:** novices only (self-rating 1–2 out of 5).
- **No personal data:** pseudonymous IDs only.
- **Consent:** a one-line consent before starting.
- **No hints:** the researcher doesn't suggest words, because a hint would
  become part of the query.
- **Names in the note:** some notes name the grape ("Barbera d'Asti"), so
  participants are asked not to repeat names. The leak check catches it if
  they do. Their words are kept as written, with no retries.
- **Which 30 wines:** chosen from the 200 test wines with at least 5 per
  variety band (`make human-cards`), so the comparison covers rare grapes too.
- **Ethics:** check IE's rules for human participants before recruiting.

**"Is n = 30 enough?"** For a validity check, yes. For a statistical test,
no. `results/llm_vs_human.csv` reports hit counts per model, not p-values.

---

## 4. Refinement — "sweeter", "less oaky", "cheaper" (D4, D5)

**The idea:**

- In embedding space, notes about sweet wines sit in a different region from
  notes about dry wines.
- The **direction vector** for a trait is the average embedding of training
  notes containing a *build* word ("sweet", "sugar", ...) minus the average of
  notes without one, scaled to length 1. It points from "not sweet" towards
  "sweet".
- **"Sweeter":** q′ = normalise(q + α·d). **"Drier":** the same with −d.
- **Search:** the Semantic model then searches with q′ instead of q.

**Leakage:** direction vectors use training notes only. α is chosen on the
dev queries, and test queries are used once, for the final numbers.

**Why two separate word lists (D5)?** If the build words also judged success,
the test would be circular. Shifting towards notes that say "sweet" would
trivially produce more notes saying "sweet". So success is measured with a
different list ("honey", "syrupy", "dessert", ...). A test checks the lists
share no word stem. Success means the shift found notes that describe the
trait in *other* words, which is a real change in results, not a word echo.

**The negation guard:** "not sweet" doesn't count as sweet, either when
building or when measuring.

**Success:** the share of top-5 notes containing a measure word goes up for
"more" and down for "less".

- **Excluded cases, counted:** cases where success is impossible are left out
  and counted (`n_eligible`). "Less oaky" can't succeed if no top-5 note
  mentions oak to begin with.

**α, the step size:**

- **The rule:** test 0.05–0.5 on dev, and choose the *smallest* α whose
  average success is within 2 points of the best
  (`results/refinement_alpha_dev.csv`).
- **Why the smallest?** A bigger step always "succeeds" more, but it drifts
  further from what the user asked in the first place. This picks the
  gentlest shift that works.
- **Warning:** if α lands at the top of the grid, the program warns, because
  the grid may be too narrow. Extending it must be decided on dev, never
  after seeing test.

**The two controls** (`results/refinement.csv`):

- **(a) Random directions:** *any* shift changes the top 5, so some "success"
  happens by chance. Ten random directions with the same α show how much.
  The trait directions must clearly beat that rate.
- **(b) The cost:** how far the originally searched wine drops in the
  ranking, and how often it leaves the top 5. Refinement is supposed to move
  away from the original, but the thesis should show how far.

**"Cheaper"** isn't a direction (D4), because price isn't in the text. The
new maximum price is the median price of the current top 5; wines without a
price are excluded. Success means the median price of the new top 5 is
strictly lower.

**"Why only the Semantic model?"** Refinement needs a vector to move. TF-IDF
and Attribute could only add filters.

**The sommelier check:** 10 test cases (fixed seed) are exported with the
query, the refinement, and the top 5 before and after
(`results/refinement_sommelier_sample.csv`). The sommelier judges whether the
wines really moved in the right direction. The word lists only measure
words, so a human has to confirm the taste really changed.

---

## 5. Weaknesses to raise yourself

1. **The LLM sees the note; humans don't.** The copy share and the human
   comparison measure how much this matters, but don't remove it.
2. **Queries describe *one wine*,** whereas real users describe what they
   want. Self-retrieval measures matching ability, not satisfaction.
3. **The vocabulary check is word-based.** "Finish" is banned even as a verb,
   and a paraphrase of expert vocabulary ("it grips your gums") passes.
4. **Llama 3.2 3B is a small model,** chosen for 8 GB of RAM. Expect more
   retries and flags; that is why both rates are reported.
5. **"Less X" = minus "more X"** assumes a trait runs along one straight line
   in embedding space. It is a simplification.
6. **Refinement success is measured on words in critics' notes,** not on how
   the wine tastes. That is why there is a sommelier check.
7. **The human subset is small** (about 30), so it is descriptive only.

---

## How the code is laid out

| Piece | One line you should be able to say |
|---|---|
| `text.normalise`, `term_pattern` | Lower case, no accents; whole words, plurals, optional negation guard. |
| `vocab.load_vocab` | Reads the two tiers the student approved. |
| `vocab.leaked_names` | Finds the wine's own grape, winery or region in a query. |
| `text.overlap_share` | How much of a query is copied from its note. |
| `llm.CachedLLM` | Asks Ollama once per prompt and stores the answer; the cache is the record. |
| `queries.write_query` | Prompt, check, feedback retry, early stop, flag. |
| `queries.quality` | Violation, retry and flag rates, length, copy share. |
| `human.select_human_wines` | 30 test wines spread over bands, shuffled. |
| `refinement.direction_vector` | Mean of notes with the trait minus mean of those without. |
| `refinement.shift` | q + α·d (or −d), scaled back to length 1. |
| `refinement.choose_alpha` | The smallest α within 2 points of the best, on dev. |
| `Refiner.evaluate` | Before/after share, eligibility, success, random control, target rank. |
| `compare_human.compare` | Hits per model for LLM vs human queries on the same wines. |
