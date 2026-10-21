# SensoryRec — Scope v2 (after supervisor meeting, 21 Oct 2026)

*Planning document, not thesis text. Items marked **[ASSUMPTION]** are my reading of the meeting notes and need your confirmation.*

---

## 1. Assessment

The supervisor's main point is that **the project is too ambitious**. Five layers (cold-start, NLP profiling, hybrid, context-aware, explainable), plus collaborative filtering, a where-to-buy API and a spirits extension, are too much for a capstone. The fix is an **agile structure**. One objective is frozen now and does not change until submission. Everything else becomes an optional increment, and **at least one** increment must be delivered.

The good news is that the 15 Oct proposal already describes this core. The change is mostly about **removing** things and **reframing** the remainder around three concepts the supervisor named: **content-based**, **sparsity** and **long tail**. Delivery moves to a **chatbot with RAG** rather than an end-to-end product.

### How I read each note

| Note | Reading | Status |
|---|---|---|
| Ambición → problema | Cut layers; scope creep is the main risk | Clear |
| Sistema agile: objetivo 1 no cambia | One frozen core objective plus increments | Clear |
| Datos → Kaggle, content-based con la description | Keep Wine Enthusiast; core = text of `description` | Clear |
| "Longterm phenomena" / LONGTAIL | **Long-tail phenomenon**: most wines and wineries have very few reviews; popularity recommenders ignore them | [ASSUMPTION] |
| SPARSITY | No user–item interactions, so CF is not viable, which justifies content-based | Clear |
| Dataset → más visual | **Bottle images** shown with each recommendation (Vivino-style), so the user can recognise the bottle in a shop | Confirmed 21 Oct |
| Foto de menú / menus on Google | Analyse restaurant wine lists (photo or online menus) | Discussed; backlog |
| Friend idea mola / 2 inputs diferentes de usuarios | **Group mode**: two friends in the app get one shared recommendation | Confirmed; late extra, low priority |
| EXTRA MÁS QUE TODO → mínimo un caso | At least **one complete worked example** in the demo, end to end | Confirmed 21 Oct |
| Map / location / availability | Show where to buy the recommended bottle; rank by what is available near the user | Discussed; backlog, feasibility unproven |
| No end-to-end, but a chatbot; RAG | Interface = conversational; retrieval + LLM-generated grounded explanation | Clear |
| Accesibilidad | Novice-friendly language (the vocabulary gap) and a usable interface | [ASSUMPTION] |

---

## 2. Frozen Objective 1 (does not change until the end)

> Build and evaluate a **content-based recommender** that matches a novice's plain-language taste description to professional tasting notes, delivered through a **RAG chatbot**. Test whether it retrieves more relevant wines than popularity, keyword (TF-IDF) and attribute-based baselines **under data sparsity**, and whether it exposes more of the catalogue's **long tail**.

### What is IN the core
- Wine Enthusiast Reviews (Kaggle), `description` as the main signal; structured fields (variety, country, price, points) as filters and baseline features
- Sentence-BERT embeddings + cosine retrieval
- Baselines: Popularity, TF-IDF, Attribute-based
- Plain-language refinement inside the chat ("less oaky", "cheaper")
- RAG: the LLM writes the explanation **only from retrieved tasting notes**
- Offline evaluation: accuracy + long-tail/beyond-accuracy metrics
- Minimal chat interface (Streamlit or Gradio)

### What is OUT (moved to Future Work)
| Removed | Why it is a defensible cut |
|---|---|
| Collaborative filtering (taster-as-user proxy) | Sparsity: the sample has only 17 tasters for 500 reviews, and ~19% of rows have no taster. A "user" matrix of ~20 rows is not CF. This *strengthens* the content-based argument |
| Hybrid layer (as CF + CB) | Nothing to hybridise without CF. "Hybrid" can survive only as semantic score + attribute filter, which should be named as such |
| Context-aware layer (CARS) | No context signal in the data; needs synthetic context, which is weak evidence |
| Wine-Searcher API / geolocation | Engineering, not research; API access and cost risk |
| Spirits | Dataset contains no spirits. **The title should drop "and Spirits"** or keep it only as future work |

---

## 3. Worked example + increments (agile backlog, in priority order)

Only start an increment once the core passes its definition of done (§6).

### Required: one complete worked example ("mínimo un caso")
One scripted scenario shown end to end in the demo and written up in the thesis:
persona (e.g. a 24-year-old novice) → plain-language query → top-5 with **bottle images** → one refinement turn ("less sweet, under €20") → grounded explanation → which bottle to look for in the shop.
This is the backbone of the defence demo. Build it as soon as the core chat works (end of S3).

### Increment A — Bottle images ("más visual")  · *priority 1*
- **Data fact:** the Kaggle dataset has **no images and no bottle identifiers** (no barcode/EAN). Images must come from elsewhere and be matched by `title` / `winery`.
- **Data fact to verify:** the dataset was scraped several years ago (check the Kaggle page and your Data Dictionary for the exact date). Many vintages will no longer be on shelves, and the catalogue is US-market-centric. This affects images *and* any availability feature.
- Options, from safest to riskiest:

| Option | Effort | Risk |
|---|---|---|
| a) **Curated demo subset**: source images manually for the ~50–100 wines used in the worked example and evaluation demo; record source and licence per image in a CSV | Low | Small coverage, but fully honest and reproducible |
| b) Image search API by title, with automated matching | Medium | API cost; wrong-bottle matches; image licensing |
| c) Scraping Vivino or similar | Medium–high | **Likely breaches the site's Terms of Service; check before doing anything.** Not defensible in a thesis |
| d) Fallback for any wine without an image: generic illustration by colour/type + the label text to look for | Low | Not a real bottle, but never wrong |

- **Recommendation:** (a) + (d). It satisfies "más visual" and the worked example without a data-acquisition sub-project. State coverage honestly ("images available for N of 129,971 wines").
- This is a **UI feature, not a research contribution**: no RQ needed, one paragraph in the methodology.

### Increment B — Group mode (two friends)  · *priority 2, late extra*
- Two users each describe their taste; the system returns one list that suits both.
- Compare **average** vs **least misery** aggregation. *Literature to source and verify: group recommender chapter in the Recommender Systems Handbook.*
- Evaluation: pair two held-out queries; report each person's target rank and the worse of the two. No new data needed.

### Increment C — Restaurant menus  · *priority 3, stretch*
- Photo of a wine list (OCR) or an online menu → rank only those wines for the user.
- **Main risk: entity matching.** Most restaurant wines will not be in Wine Enthusiast. Fallback: proxy profile from variety + region, clearly labelled as an estimate.
- **Online menus via Google:** check Google's Terms of Service and API limits before planning this. Collecting menus by scraping is not defensible. Manually collected photos (10–20) are enough for a demonstrator.

### Increment D — Map, location and availability  · *backlog / Future Work*
**This is where the ambition problem comes back, so be careful.**
- The recommender itself can't use location: **there is no availability data** in the dataset. A map of shops (e.g. from a places API) shows *where shops are*, not *whether they stock this bottle*.
- Real availability needs a stock source: a retailer API, a price-comparison API (paid, access not guaranteed), or a partner such as LVMH Moët Hennessy or a local retailer sharing a product list.
- Combined with the old, US-centric catalogue, a "best bottle available near you" feature would mostly return nothing in Madrid.
- **Recommendation:** put it in **Future Work** as the business extension (it is a strong point for the business-implications chapter). At most, show a **clearly labelled mock-up** in the demo. Only reconsider it if a legitimate stock data source appears.

---

## 4. Research questions v2

**Main RQ.** For users with no interaction history, does semantic matching of plain-language taste descriptions to expert tasting notes retrieve more relevant wines than popularity, keyword and attribute-based recommenders?

| Sub-RQ | Hypothesis *(not yet tested)* | Metric |
|---|---|---|
| SQ1 — Long tail: does the semantic recommender expose more of the long tail without losing accuracy? | H1: higher catalogue coverage and tail share than Popularity, at equal or better HR@5 | Catalogue coverage, % recommendations from tail items, Gini index, novelty |
| SQ2 — Refinement: do plain-language critiques move results in the requested direction? | H2: direction-vector shifts move results toward the trait in a majority of cases | Directional success rate |
| SQ3 — Grounding: are RAG explanations faithful to the retrieved notes? | H3: grounded prompting keeps unsupported claims rare | % explanations with an unsupported claim (manual annotation, ~50 samples) |
| SQ-ext — Group (if Increment B) | H4: least misery gives a better worst-member rank than average | Worst-member rank, mean rank |

**Change vs. 15 Oct proposal:** the old five sub-RQs included CF and context. Those are gone. SQ1 and SQ3 are new and reflect "long tail" and "RAG".

### Define "long tail" before measuring it
Choose and justify one operational definition, e.g. *tail = wines from wineries/varieties in the bottom 80% by review count*. Sample evidence (500 rows only; **must be re-computed on the full 129,971**): 473 wineries for 500 reviews, 90 varieties, the top 5 varieties are 38% of rows, and 46 varieties appear once. That is a strong long-tail pattern, but the full-dataset numbers go in the thesis, not these.

---

## 5. Evaluation plan (updated)

| What | Method | Notes |
|---|---|---|
| Accuracy | Self-retrieval: ~200 held-out notes rewritten as casual queries; HR@5, NDCG@5 | Already in proposal |
| Long tail | Coverage, tail share, Gini, novelty across all test queries | New |
| Diversity | Intra-list diversity (1 − mean pairwise cosine) | Cheap, add |
| Refinement | Directional success rate | Already in proposal |
| Grounding | Manual annotation of ~50 RAG explanations | New |
| Expert check | Sommelier rates top-5 for 30 queries (P@5) | Confirm sommelier in writing |
| Accessibility | Small novice check (5–8 people): could they use it without wine vocabulary? Short SUS-style questionnaire | New, light; check IE ethics requirements for human participants |

**Critical methodological risks**
1. **Circularity in casual queries.** If an LLM writes the casual paraphrases, it may reuse tasting-note vocabulary and favour TF-IDF/SBERT unfairly, or produce text no novice would write. Fix: fixed prompt that bans expert terms, automatic check against a wine-vocabulary list, and a subset of queries written by real novices.
2. **LLM inside evaluation.** Keep the LLM **out of the ranking**. Rankings must be deterministic (embeddings + scores); the LLM only writes explanations. This keeps the main results reproducible.
3. **LLM reproducibility and cost.** Temperature 0, pinned model version, cache all generated outputs to disk and commit them. A local open model is an alternative if API access is a problem.
4. **Popularity bias in "relevance".** Points/price correlate with what the sommelier may prefer; report this as a limitation.

---

## 6. Agile structure

| Sprint (2–3 wks) | Goal | Definition of done |
|---|---|---|
| S0 | Repo, data pipeline, full-dataset EDA incl. long-tail stats | `make data` reproduces cleaned parquet + EDA figures |
| S1 | Baselines + SBERT retrieval + metrics module | Ablation table (4 models × HR@5/NDCG@5/coverage) generated by one command |
| S2 | Casual query set + refinement | 200 validated queries; directional success computed |
| S3 | RAG chatbot (core frozen here) + worked example | Chat answers with top-5 + grounded explanation; worked example runs end to end; faithfulness annotated |
| S4 | Increment A (bottle images) | Images for the demo subset + fallback; sources/licences logged |
| S5 | Increment B (group mode) | Aggregation comparison table |
| S6 | Increment C (menus) — stretch | Demo on 10–20 photos with reported match rate |
| S7 | Sommelier session, novice check, write-up | — |

Map sprints to the existing deadlines: Literature Review 25 Jan, Methodology 20 Feb, Full Draft 20 Mar, Final 10 Apr. **S0–S3 should be done before 20 Feb** so the methodology chapter describes a working core.

---

## 7. Plan to the next meeting — Monday 2 November 2026 (12 days)

**Goal for the meeting:** show that the scope is now controlled, and that the core has started working.

| By | What | Output to bring |
|---|---|---|
| Fri 23 Oct | Write Objective 1, IN/OUT table and increment priority as a one-page summary in your own words | 1-page scope v2 |
| Wed 28 Oct | **S0 done:** full-dataset cleaning + EDA, incl. long-tail stats (reviews per winery/variety, Lorenz curve) and the dataset's scrape date | EDA figures; long-tail numbers on the full 129,971 rows |
| Sun 1 Nov | **S1 started:** Popularity + SBERT retrieval running on a few example queries; metrics module with unit tests | 3–5 live example queries; first draft of the ablation table if ready (no numbers unless they come from the code) |
| Sun 1 Nov | Storyboard of the worked example, including where bottle images appear | 1 slide or sketch |

**Questions to settle with the supervisor on 2 Nov**
1. Is the priority order (images → group → menus → map in Future Work) acceptable?
2. For images: is a curated demo subset enough, or do they expect broad coverage?
3. Is a labelled map mock-up in the demo acceptable, or should location be left entirely to Future Work?
4. Updated title: drop "and Spirits" (no spirits in the data); keep or drop "Hybrid"?

**Other next steps**
- Literature gaps created by the new scope, all **to be sourced and verified**: long-tail/popularity bias; RAG and faithfulness evaluation; conversational recommenders (most critical gap); group recommender aggregation (later).
- Start Claude Code on S0–S1 using `CLAUDE.md`.

*AI-use note: drafted by Claude (Anthropic), 21 Oct 2026. Planning material, not for submission as written.*
