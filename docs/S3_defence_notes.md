# S3 — Defence notes (grounded explanations and the worked example)

*Study notes for the oral defence, not thesis text. Every number comes from
a results file; this note names the file and the column instead of copying
values (CLAUDE.md rule 3). The plan and its alternatives are in
`DECISIONS.md` (entry of 19 Nov 2026, D1–D4).*

Files behind this note:

| What | File |
|---|---|
| Explanation code (prompt building, parsing, quote check, fallback) | `src/sensoryrec/rag.py` |
| Worked example code | `src/sensoryrec/worked_example.py` |
| Prompt, model, persona, budget, refinement | `config.yaml`, section `rag` |
| Worked example, readable / with pictures / one row per wine | `results/worked_example.md`, `.html`, `.csv` |
| 50 explanations for annotation, with the automatic quote check | `results/faithfulness_sample.csv` |
| Every LLM answer | `data/llm_cache/` |
| Cloud run | workflow "Explanations" (push to `run-rag`), output branch `rag-results` |

---

## The 60-second version

> "The recommender ranks the wines; the language model only explains them.
> After the engine picks the top five, I send the model the person's request
> and, for each wine, only its name, grape, country, price and tasting note,
> in the engine's order. The model must write one sentence per wine, and each
> sentence must quote a few words of that wine's note, so every claim can be
> traced back to the source. It may not reorder, add or drop wines, and it
> must say so when a note does not cover what the person asked for. I check
> faithfulness twice: a script tests whether every quoted phrase really
> occurs in the note, and I read 50 explanations by hand and mark each one
> faithful, partly faithful or not, writing down any claim that is not in the
> notes. If the model cannot answer, the app shows the words the request
> shares with the note, clearly labelled, so it never shows an invented
> explanation."

---

## 1. Where the LLM sits, and where it does not

Retrieval-augmented generation in its simplest form:

1. **Retrieval.** The engine (`rag.engine`, Sentence-BERT by default) ranks
   the catalogue with cosine similarity, plus the budget filter and the
   refinement shift. Fully deterministic (rule 2).
2. **Augmentation.** `describe()` turns the top k into numbered lines:
   title, grape, country, price, tasting note. Nothing else, so no points
   score, no winery facts, no internal ids (a test checks this).
3. **Generation.** Gemini Flash-Lite writes JSON: one intro sentence and one
   sentence per wine. `parse()` refuses an answer that misses a wine or the
   intro; then the fallback is shown instead.

**Likely question:** *"Can the LLM change the recommendation?"* → No. The
order shown in the app is the engine's; the LLM's answer is matched to the
wines by number (`"wine": 1..k`) and only supplies the text next to each one.

## 2. The prompt, line by line (`config.yaml`, `rag.prompt`)

- "A search engine has already chosen these k wines, in this order. Do not
  change the order…" — keeps the LLM out of ranking.
- "Use only the information above. Do not add facts about the winery, the
  region, food or prices" — the grounding rule.
- "Each sentence must contain a short exact quote (3 to 8 words)" — makes
  every explanation checkable by a script and by a reader.
- "If the note does not cover a wish… say so honestly" — prevents the model
  from pretending a dry wine is sweet. The worked example shows this
  happening (turn 1).
- "Explain any wine word a beginner may not know" — the audience is a
  novice. This line is also the source of most extra content (see §5).
- JSON only, `response_mime_type: application/json` — so the answer can be
  parsed reliably.

Temperature 0, fixed seed, model pinned (`gemini-3.1-flash-lite`), every
answer cached by prompt hash. Why Flash-Lite and not the local Llama 3.2 3B:
better instruction following and the app does not need Ollama running
(DECISIONS, 19 Nov, D1).

## 3. The refinement turn

When the person clicks "sweeter" (or "cheaper"), the engine re-ranks; then
`change_text()` adds to the prompt what was asked for and which positions
hold wines that were not in the previous list. The intro must say in a few
words what changed. The LLM is told what changed; it does not decide it.

## 4. The worked example (`make worked-example`)

Required by CLAUDE.md: persona → query → top 5 with bottle images →
refinement → grounded explanation, end to end from one command.

- Persona Lucía, 24, new to wine; request "Something fruity and not too dry
  to drink with friends over dinner, easy to like"; budget filter $20;
  refinement "sweeter". The request passes the expert-vocabulary check.
- Pictures: a real photo where one is logged in `data/images/sources.csv`,
  otherwise the drawn bottle by grape and colour.
- The script stops with an error if any turn fell back, so the committed
  example always has real explanations.

**What it shows, honestly (read it in `results/worked_example.md`):**

- In turn 1 some wines are described in their notes as *dry*, although the
  person said "not too dry". Sentence-BERT puts "not too dry" close to
  "dry": embeddings handle negation poorly. This is a known weakness, worth
  stating before the examiner finds it.
- The explanation does not hide it: it quotes the note and says the wine is
  dry. That is the grounding rule working.
- After "sweeter" the list moves to wines whose notes say sweet, still under
  the budget. That is the refinement direction doing what the plain-language
  request alone did not.

## 5. Checking faithfulness

**Automatic check** (`quote_check`): finds every double-quoted phrase of two
or more words and tests whether it occurs word for word in that wine's note
(case and punctuation ignored). Columns `quotes` and
`quotes_found_in_notes`; the run's log line gives the totals. A miss can be
harmless: the model sometimes shortens a quote with "…", which then no
longer matches word for word (case 22).

**What the script cannot see:** claims *outside* the quotes. Two kinds seen
so far:
- an added fact, e.g. explaining that a rosé is "a pink wine made from red
  grapes" (true in general, but not in the note);
- a beginner's definition of a wine word, e.g. what acidity means. The
  prompt asks for these.

**Hand annotation** (all 50, by me; D4). Per row: `faithful_yes_partly_no`
and `claims_not_in_notes`. Rule I apply:
- *yes* — every claim about the wine is in its note;
- *partly* — mostly grounded, but at least one claim about the wine is not
  in the note (write it down);
- *no* — a claim contradicts the note, or the sentence describes the wrong
  wine.
- Definitions of general wine words are not counted as claims about the
  wine, because the prompt asks for them; a wrong one is written in
  `claims_not_in_notes` but does not change the label (DECISIONS,
  19 Nov 2026, "Faithfulness annotation rule").
- Why the rosé example is *partly* and not a gloss: it states how this
  wine was made, which the note does not say.

The faithfulness rate goes in the thesis only once it is computed by a
script from the annotated sheet; until then it is "not yet available".

## 6. When the LLM cannot answer

No key, quota used up, network error or an unreadable answer → `explain()`
returns the fallback: for each wine, the words the request shares with the
note, or "matched on the meaning of the note" if there are none. The app
labels it as a fallback. `make faithfulness` and `make worked-example` stop
with an error if any explanation fell back, so committed results never mix
the two.

## 7. Reproducibility

- `make worked-example` and `make faithfulness` read every answer from
  `data/llm_cache/`: run again, no request is sent, identical files.
- The 50 requests are drawn from the usable simulated queries with the seed
  (`rag.n_faithfulness`), sorted by query id.
- Tests: `tests/test_rag.py` (prompt content and order, parsing, quote
  check, fallback, cache) and `tests/test_worked_example.py` (budget filter,
  refinement re-ranking, page contents), with fake models and a fake LLM.

## 8. Questions to expect

- *"Isn't the LLM just making things up?"* → It may, which is why each
  sentence must quote the note, a script checks the quotes, and I annotated
  50 by hand. The remaining risk is claims outside the quotes (§5).
- *"Why not let the LLM choose the wines? It understands 'not too dry'."*
  → Then the result is not reproducible or measurable, and it could
  recommend wines that do not exist in the catalogue. Rule 2 keeps ranking
  deterministic; the negation weakness is reported, and the refinement
  buttons are the deterministic way to correct it.
- *"Why only 50?"* → Hand annotation is the bottleneck; 50 is enough to
  find the kinds of error, not to give a tight rate. Say this.
- *"The same model family is your judge."* → The judge scored the
  rankings, not the explanations; the explanations are scored by me.
- *"Why Sentence-BERT in the app when the judge preferred bge-small?"* →
  The engine is one setting (`rag.engine`); the choice is open until the
  Gemini sub-catalogue comparison is in (DECISIONS, 19 Nov, D2).
