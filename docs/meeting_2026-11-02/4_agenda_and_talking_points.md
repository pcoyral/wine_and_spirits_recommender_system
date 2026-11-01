# Meeting 2 Nov 2026: agenda, talking points, questions

*For you, not for the supervisor. About 25 to 30 minutes.*

## Agenda

| Min | Topic | Show |
|---|---|---|
| 0-3 | Where I am, in one sentence | |
| 3-7 | Scope is under control | `1_scope_summary.md` |
| 7-12 | S0: the data and the long tail | 3 figures in `results/eda/` |
| 12-17 | S1: the core works (live demo) | `results/demo/demo_top5.md` (or run `make demo`) |
| 17-20 | S2: how the evaluation will work | `DECISIONS.md`, `docs/human_query_protocol.md` |
| 20-23 | The worked example | `3_storyboard.md` |
| 23-30 | Questions for him, next steps | List below |

## Talking points

Opening, in one sentence: "The scope is frozen as we agreed, the data pipeline
and all four recommenders are built and tested, and today I can show Popularity
and the semantic model side by side on real queries."

Scope: walk through the in/out table and point out what was cut (collaborative
filtering, context, maps, spirits). Use the data to explain why CF is out (next
point).

S0, the data:
- `fig_rank_frequency.png` and `fig_lorenz.png`: "Varieties are extremely
  concentrated, with a Gini of 0.93, and the top 1% of grapes hold half the
  reviews. Wineries have a long, flat tail where most have only a few reviews."
- `fig_taster_coverage.png`: "Each wine has one review by one of 19 critics,
  and a fifth have no critic. No two users rated the same wine, so CF is
  impossible. That's the sparsity argument for going content-based."
- Tail definition: "A wine is in the long tail if its winery has 10 reviews or
  fewer. On this data that lands right next to the 80/20 cut, and it's the same
  unit the Popularity baseline counts."

S1, the demo (open `demo_top5.md`):
- "Popularity gives the same wines for every query, here the most-reviewed
  winery, which is Portuguese. The semantic model follows what the person
  asked for."
- "This is qualitative. The real numbers will come from 200 held-out test
  queries."
- If he asks how it ranks: sentence embeddings and cosine similarity, no LLM.

S2, evaluation design:
- "A local LLM rewrites each held-out wine's note as a casual query, and the
  model has to find the original wine."
- "The risk is that the LLM copies the critic's words. So there are 85 banned
  expert terms, a check that the query doesn't name the wine, a measure of how
  much it copies, and about 30 queries written by real novices as a check."
- "Refinement is scored with a separate word list and compared against random
  directions, so success isn't circular."
- Open `DECISIONS.md`: "Every choice is dated and justified, and that becomes
  my methodology chapter."

AI use (bring it up before he asks): "I built the code with Claude Code, and
every commit is marked. I made every methodological decision, they're logged in
DECISIONS.md, and I've written notes on how each part works so I can defend it."
Check IE's policy before the meeting if you can.

## Questions to ask him

From the scope plan:

1. Is the increment order acceptable (images, then group mode, then menus), with
   maps left as future work?
2. Images: is a curated demo subset of 50 to 100 wines enough?
3. Is a clearly labelled map mock-up acceptable in the demo, or should location
   stay out?
4. Title: drop "and Spirits"? Keep or drop "Hybrid"?

New since then:

5. Is it acceptable for an LLM to write the 300 evaluation queries (200 test,
   100 dev), given the checks and the 30 or so human-written queries as
   validation?
6. Ethics: what does IE require before I collect the human queries (approval,
   an exemption, consent wording)? Who should I ask?
7. Sommelier: can we confirm the arrangement in writing? Is 10 refinement cases
   plus top-5 ratings for 30 queries a reasonable request?
8. Data and repo: is a public GitHub repo fine? The Kaggle data is CC BY-NC-SA
   (attribution, non-commercial). My copy came from a mirror, and I'll check it
   against Kaggle by checksum.
9. AI assistance: do I need to document anything beyond the commit trailers and
   a statement in the thesis?

## Questions he may ask you (short answers)

| Question | Answer |
|---|---|
| "Why not collaborative filtering?" | One review per wine, 19 critics, a fifth anonymous. Users never overlap, so there's no similarity to learn. |
| "Isn't Popularity a straw man?" | It's the standard non-personalised baseline and what a novice gets in a shop. SQ1 also needs a reference that concentrates on the head. TF-IDF is the real test. |
| "How do you avoid leakage?" | The 200 test wines were held out before anything was fitted, and nothing is tuned on them. Step sizes are tuned on 100 separate dev wines. |
| "Why should LLM queries be trusted?" | They're checked automatically, failures are excluded and counted, copying is measured, and about 30 human queries check realism. |
| "Is one correct wine per query too strict?" | For absolute scores, yes, which is why the sommelier rates top-5 lists. For comparing models it's fair, since all of them are judged on the same targets. |
| "What if the semantic model doesn't win?" | Then that's the finding. The ablation shows which capability (facts, words or meaning) helps, and the long-tail metrics still answer SQ1. |
| "Where are the results?" | The pipeline is ready. Generating the 300 queries with the local LLM is the next step, and then `make eval` produces the table in one command. |

## Next steps to propose (until the next meeting)

1. Generate and check the 300 queries, then run the first ablation table.
2. Get ethics clearance, then collect the human queries.
3. Start S3: the chat app with grounded explanations and the worked example.
4. Fill the literature review gaps on long tail and popularity bias, RAG
   faithfulness, and conversational recommenders (deadline 25 Jan).
