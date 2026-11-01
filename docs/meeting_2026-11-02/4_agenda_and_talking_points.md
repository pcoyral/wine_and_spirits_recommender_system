# Meeting 2 Nov 2026 — agenda, talking points, questions

*For you, not for the supervisor. About 25–30 minutes.*

## Agenda

| Min | Topic | Show |
|---|---|---|
| 0–3 | Where I am in one sentence | — |
| 3–7 | Scope is controlled | `1_scope_summary.md` |
| 7–12 | S0: the data and the long tail | 3 figures in `results/eda/` |
| 12–17 | S1: the core works — live demo | `results/demo/demo_top5.md` (or run `make demo`) |
| 17–20 | S2: how the evaluation will be done | `DECISIONS.md`, `docs/human_query_protocol.md` |
| 20–23 | The worked example | `3_storyboard.md` |
| 23–30 | Questions for him; next steps | list below |

## Talking points

**Opening (one sentence).** "The scope is frozen as we agreed, the data
pipeline and all four recommenders are built and tested, and today I can show
Popularity and the semantic model side by side on real queries."

**Scope.** Walk through the IN/OUT table. Point out what was cut: collaborative
filtering, context, maps, spirits. Say why CF is out using the data (next point).

**S0 — the data.**
- `fig_rank_frequency.png` + `fig_lorenz.png`: "Varieties are extremely
  concentrated — Gini 0.93; the top 1% of grapes hold half the reviews. Wineries
  have a long flat tail: most have only a few reviews."
- `fig_taster_coverage.png`: "Each wine has one review by one of 19 critics, and
  a fifth have no critic. No two users rated the same wine, so CF is impossible;
  that's the sparsity argument for content-based."
- Tail definition: "A wine is in the long tail if its winery has ≤ 10 reviews.
  It sits right next to the 80/20 cut on this data, and it's the same unit the Popularity
  baseline counts."

**S1 — the demo.** Open `demo_top5.md`.
- "Popularity gives the same wines for every query — here the most-reviewed
  winery, from Portugal. The semantic model follows what the person asked."
- "This is qualitative. The real numbers come from 200 held-out test queries."
- If he asks how it ranks: sentence embeddings, cosine similarity, no LLM.

**S2 — evaluation design.**
- "Each held-out wine's note is rewritten as a casual query by a local LLM; the
  model must find the original wine."
- "The risk is that the LLM copies the critic's words, so: 85 banned expert
  terms, a check that the query doesn't name the wine, a measure of how much it
  copies, and ~30 queries written by real novices as a check."
- "Refinement is measured with a separate word list and against random
  directions, so success isn't circular."
- Open `DECISIONS.md`: "Every choice is dated and justified; that becomes my
  methodology chapter."

**AI use (say it before he asks).** "I built the code with Claude Code — every
commit is marked. I made every methodological decision, it's logged in
DECISIONS.md, and I've written notes on how each part works so I can defend it."
Check IE's policy before the meeting if you can.

## Questions to ask him

From the scope plan:

1. Is the increment order acceptable — images → group mode → menus, maps as Future Work?
2. Images: is a curated demo subset (~50–100 wines) enough?
3. Is a clearly labelled map mock-up acceptable in the demo, or leave location out?
4. Title: drop "and Spirits"? Keep or drop "Hybrid"?

New since then:

5. Is it acceptable that an LLM writes the 300 test queries, given the checks
   and the ~30 human-written queries as validation?
6. **Ethics:** what does IE require before collecting the ~30 human queries —
   approval, an exemption, a consent wording? Who do I ask?
7. Sommelier: can we confirm in writing? Is 10 refinement cases + top-5 ratings
   for 30 queries a reasonable request?
8. Data and repo: is a public GitHub repo fine? The Kaggle data is CC BY-NC-SA
   (attribution, non-commercial); the copy I used came from a mirror and will be
   checked against Kaggle by checksum.
9. AI assistance: anything I must document beyond the commit trailers and a
   statement in the thesis?

## Questions he may ask you (short answers)

| Question | Answer |
|---|---|
| "Why not collaborative filtering?" | One review per wine, 19 critics, a fifth anonymous: no overlap between users, so no similarity to learn. |
| "Isn't Popularity a straw man?" | It's the standard non-personalised baseline and what a novice gets in a shop; SQ1 needs a model that concentrates on the head. TF-IDF is the real test. |
| "How do you avoid leakage?" | 200 test wines held out before anything is fitted; nothing is tuned on them; step sizes are tuned on 100 separate dev wines. |
| "Why should LLM queries be trusted?" | They're checked automatically, failures are excluded and counted, copying is measured, and ~30 human queries check realism. |
| "Is one correct wine per query too strict?" | Yes for absolute scores, which is why the sommelier rates top-5 lists; for comparing models it's fair, since all are judged on the same targets. |
| "What if the semantic model doesn't win?" | Then that's the finding; the ablation shows which capability (facts, words, meaning) helps, and the long-tail metrics still answer SQ1. |
| "Where are the results?" | The pipeline is ready; the 300 test queries are the next step (local LLM), then `make eval` produces the table in one command. |

## Next steps to propose (until the next meeting)

1. Generate and check the 300 queries; run the first ablation table.
2. Ethics clearance → collect the human queries.
3. Start S3: the chat app with grounded explanations and the worked example.
4. Literature review gaps: long tail / popularity bias, RAG faithfulness,
   conversational recommenders (deadline 25 Jan).
