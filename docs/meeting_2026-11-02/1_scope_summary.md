# SensoryRec — Scope v2 in one page

**Paul — IE University BDBA Capstone · supervisor meeting 2 Nov 2026**

## Frozen objective 1 (fixed until submission)

Build and evaluate a content-based recommender that matches a novice's
plain-language description of what they like to professional tasting notes
(Wine Enthusiast, Kaggle), delivered through a RAG chatbot. Test whether it
finds more relevant wines than popularity, keyword (TF-IDF) and attribute
baselines under data sparsity, and whether it surfaces more of the
catalogue's long tail.

## Why content-based

There are no user-item interactions to learn from. Each wine has exactly one
review, written by one of 19 critics, and about a fifth of the reviews name no
critic at all. Since no two "users" ever rated the same wine, collaborative
filtering has nothing to work with. The tasting notes are the signal.

## In / out of scope

| In (core) | Out (future work) |
|---|---|
| Data pipeline and EDA on the full dataset | Collaborative filtering |
| 4 recommenders: popularity, TF-IDF, attribute, semantic (Sentence-BERT) | Hybrid CF + content |
| Plain-language refinement in the chat ("less oaky", "cheaper") | Context-aware recommendation |
| RAG: the LLM explains from the retrieved notes and never ranks | Location, availability, maps (at most a labelled mock-up) |
| Offline evaluation: accuracy, long tail, diversity | Purchase APIs, spirits, model fine-tuning |
| Minimal chat interface with one complete worked example | |

## Increments (after the core works, in this order)

1. A. Bottle images for a curated demo subset, with a generic fallback.
2. B. Group mode: two friends get one shared list.
3. C. Restaurant wine list (stretch): rank the wines from a photo of the menu.

## Research questions

- Main: for users with no history, does semantic matching of plain language
  beat popularity, keyword and attribute recommenders?
- SQ1 (long tail): does it recommend more of the tail without losing accuracy?
- SQ2 (refinement): do plain-language critiques move results in the requested
  direction?
- SQ3 (grounding): are the chatbot's explanations faithful to the notes?

## Evaluation

Held-out tasting notes are rewritten as casual queries, and each model has to
find the original wine (HR@5, NDCG@5). On top of accuracy we report coverage,
tail share, Gini, novelty and diversity. A small set of human-written queries,
a sommelier check and a faithfulness annotation validate the automatic results.

## Timeline

Core (S0 to S3) done before the methodology deadline on 20 Feb; increments
come after.

- Literature review: 25 Jan
- Methodology: 20 Feb
- Full draft: 20 Mar
- Final: 10 Apr
