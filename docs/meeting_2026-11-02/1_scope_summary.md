# SensoryRec — Scope v2 in one page

> **DRAFT by Claude. Rewrite it in your own words before the meeting** (the
> plan for this meeting asks for the summary "in your own words"). Keep the
> structure; change the wording.

**Paul — IE University BDBA Capstone · supervisor meeting 2 Nov 2026**

## Frozen Objective 1 (does not change until submission)

Build and evaluate a **content-based recommender** that matches a novice's
plain-language description of what they like to professional tasting notes
(Wine Enthusiast, Kaggle), delivered through a **RAG chatbot**. Test whether it
finds more relevant wines than popularity, keyword (TF-IDF) and attribute
baselines **under data sparsity**, and whether it exposes more of the
catalogue's **long tail**.

## Why content-based

There are no user–item interactions to learn from. Each wine has exactly one
review, written by one of 19 critics (and about a fifth have no critic named),
so no two "users" ever rated the same wine. Collaborative filtering has
nothing to work with; the text of the tasting notes is the signal.

## IN / OUT

| IN (core) | OUT (Future Work) |
|---|---|
| Data pipeline + EDA on the full dataset | Collaborative filtering |
| 4 recommenders: Popularity, TF-IDF, Attribute, Semantic (Sentence-BERT) | "Hybrid" CF + content |
| Plain-language refinement in the chat ("less oaky", "cheaper") | Context-aware recommendation |
| RAG: the LLM only *explains*, from the retrieved notes; it never ranks | Location, availability, maps (at most a labelled mock-up) |
| Offline evaluation: accuracy + long-tail + diversity | Purchase APIs, spirits, fine-tuning models |
| Minimal chat interface + **one complete worked example** | |

## Increments (only after the core works, in this order)

1. **A — Bottle images** for a curated demo subset, with a generic fallback.
2. **B — Group mode:** two friends, one shared list.
3. **C — Restaurant wine list** photo → rank the menu's wines (stretch).

## Research questions

- **Main:** for users with no history, does semantic matching of plain
  language beat popularity, keyword and attribute recommenders?
- **SQ1 — Long tail:** does it recommend more of the tail without losing accuracy?
- **SQ2 — Refinement:** do plain-language critiques move results the requested way?
- **SQ3 — Grounding:** are the chatbot's explanations faithful to the notes?

## Evaluation in one line

Held-out tasting notes are rewritten as casual queries; each model must find
the original wine (HR@5, NDCG@5), plus coverage, tail share, Gini, novelty and
diversity; a small human-written subset, a sommelier check and a faithfulness
annotation validate the automatic results.

## Timeline

S0–S3 (core) done before the methodology deadline (20 Feb); increments after.
Literature review 25 Jan · Methodology 20 Feb · Full draft 20 Mar · Final 10 Apr.
