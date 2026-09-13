# SensoryRec — Proposed Optimal Academic Scope
**Purpose:** A grade-optimized, defensible scope for the IE BDBA Capstone, synthesized from the project brief, data strategy, feature engineering strategy, EDA, and literature already gathered.
**Design principle:** Go deep on the part of the project that is genuinely novel (natural-language preference elicitation for cold-start wine recommendation), treat collaborative filtering and context-awareness honestly given data constraints, and keep the broader 5-layer vision visible as a roadmap rather than a build commitment.

---

## 1. Final Project Title

> **SensoryRec: Bridging the Vocabulary Gap — A Hybrid Content-Based and Natural-Language Recommender System for Cold-Start Wine Discovery**

**Why this title:**
- Keeps the established "SensoryRec" identity.
- "Bridging the Vocabulary Gap" names the precise consumer-behaviour problem (Label Fluency / self-confidence literature, plus the L.E.K. 2026 finding that "improving my ability to find wines I enjoy" is the top driver of Gen Z nonconsumer interest).
- "Hybrid Content-Based and Natural-Language" accurately describes the scoped methodology (Section 6) without overclaiming a fully personalized collaborative system.
- "Cold-Start" signals the academic framing examiners will recognize from the course (S21) and positions the contribution precisely: this is fundamentally a cold-start problem, and natural language is the proposed solution.
- Drops "Wine and Spirits" and "Discovery" generality from the original title — spirits becomes future work (Section 9), keeping the title's promise matched to what's actually delivered.

---

## 2. Main Research Question

> **RQ:** *To what extent can natural-language descriptions of taste preferences — requiring no wine-specific vocabulary — be used to generate wine recommendations whose relevance, as measured by offline ranking metrics and validated by an expert sommelier, matches or exceeds that of a structured attribute-based content recommender, for users with no prior interaction history?*

This single sentence does the work of: (a) defining the independent variable (natural-language input vs. structured attributes), (b) defining the dependent variable (relevance, operationalized via specific metrics in Section 7), (c) specifying the population (cold-start/novice users), and (d) committing to expert validation as part of the evidentiary standard — all of which examiners will look for explicitly.

---

## 3. Sub-Research Questions

1. **(Mechanism design)** What computational method can effectively map informal, free-text descriptions of taste preference onto the SBERT-derived semantic embedding space of professional wine tasting notes, and how can comparative refinements (e.g., "more fruity," "less oaky") be operationalized as transformations within that embedding space?

2. **(Core comparative evaluation)** How does an NLP-embedding-driven recommender perform — in terms of Precision@K, Recall@K, and NDCG@K against a sommelier-validated relevance set — relative to (a) a non-personalized popularity baseline and (b) a structured attribute-based content recommender (variety, region, price, points)?

3. **(Iterative refinement — primary novelty)** Does allowing users to iteratively refine recommendations through natural-language feedback (e.g., "less sweet, more like the second one") improve top-K relevance and user-perceived control relative to a single-shot natural-language query?

4. **(Beyond-accuracy quality)** How do the NLP-driven and attribute-based approaches compare on catalog coverage, intra-list diversity, and serendipity — and does optimizing for linguistic accessibility come at a cost to recommendation diversity?

5. **(Path to personalization)** Using professional tasters' review histories within the Wine Enthusiast dataset as a proxy for individual "user" taste profiles, can collaborative-filtering signals be meaningfully combined with the NLP-driven content layer — and what does this reveal about the feasibility of a fully personalized, returning-user version of SensoryRec?

**Mapping to the report:** RQ1 → Methodology design chapter; RQ2 & RQ4 → core Results chapter (the "ablation table"); RQ3 → the headline novel-contribution result; RQ5 → a clearly-scoped exploratory section that honestly addresses the collaborative-filtering question without relying on fabricated synthetic users.

---

## 4. Expected Academic Contribution

- **A reproducible method** for translating informal natural-language taste descriptions into adjustments within a professional-review-derived embedding space — operationalizing a problem (the "vocabulary gap") that the consumer-behaviour literature (Label Fluency and Consumer Self-Confidence; L.E.K. 2026) identifies descriptively but does not address computationally.
- **An empirical ablation study** (non-personalized → attribute-based content → NLP-embedding → NLP + iterative refinement) under a single, consistent evaluation protocol — directly extending the recommended next step from the project's own EDA ("build a baseline content-based recommender, then compare with TF-IDF/SBERT embeddings") into a full hybrid comparison.
- **A novel application of the cold-start framing**: most cold-start literature (S21) addresses the problem via onboarding questionnaires, demographic priors, or hybrid switching; SensoryRec proposes natural language *itself* as the cold-start solution mechanism — a contribution at the intersection of CARS/hybrid-RS theory and NLP.
- **An honest, methodologically interesting treatment of the "no interaction data" constraint**, using professional tasters' rating histories as a documented, real (if imperfect) proxy for collaborative signals — turning a stated dataset limitation into a small but genuine empirical exploration (RQ5), rather than relying on synthetic/simulated users.
- **A reproducible, version-controlled pipeline** (GitHub) from raw Wine Enthusiast data through feature engineering, embedding generation, model comparison, and demo deployment — satisfying the reproducibility commitment in the Data Strategy document.

---

## 5. Expected Business Contribution

- **A direct response to the L.E.K. 2026 finding** that the single biggest unmet need among Gen Z nonconsumers and light consumers is *"improving my ability to find wines I enjoy"* — SensoryRec's natural-language interface is positioned as exactly this capability.
- **A working prototype** (Section 9) demonstrating that LVMH Moët Hennessy could offer a "describe what you like, in your own words" discovery tool — directly relevant to digital/DTC channels and consistent with the premiumization trend (premium bottles ≥$15 showing relative resilience even as overall category volume declines).
- **A documented roadmap from prototype to product**: the report's "Future Work" section (informed by the de-scoped Layers 3–5 and the spirits extension) becomes a credible extension plan — e.g., integrating real interaction data (Vivino, retail POS) to upgrade the proxy collaborative-filtering exploration (RQ5) into a true returning-user system, and adding context-aware filtering (occasion, time of day) as outlined in S18/S19.
- **A lower barrier to entry framing** that aligns with the L.E.K. report's recommendation to "simplify messaging and invest in consumer education" — SensoryRec operationalizes that recommendation as a product feature rather than a marketing claim.
- **A sommelier-validated proof of credibility**: expert sign-off on recommendation quality gives LVMH stakeholders a non-technical, trust-based argument for the system's plausibility, independent of the academic metrics.

---

## 6. Final Methodology

### 6.1 Data & Feature Engineering (as already scoped, lightly extended)
- Source: Wine Enthusiast Reviews (129,971 rows, 13 columns), per the existing EDA and Data Dictionary.
- Cleaning: handle missingness as already characterized (region_2 61%, designation 29%, taster_name 20%, region_1 16%, price 7%) via documented imputation/exclusion rules.
- Features (per Feature Engineering Strategy): SBERT (`all-MiniLM-L6-v2`) embeddings of `description` (384-d); log-transformed `price`; min-max scaled `points`; one-hot `variety` and `country`; encoded `region_1`/`region_2`; composite "wine vector" = embedding ⊕ price ⊕ rating ⊕ categorical features.

### 6.2 Model Ladder (the ablation backbone)
Build and evaluate, in increasing order of sophistication — each model directly answers one sub-RQ:

1. **Model 0 — Non-Personalized Baseline** (per S09): top-rated / most-popular wines, optionally filtered by price band. Establishes the floor.
2. **Model 1 — Attribute-Based Content Filtering**: cosine similarity over the structured portion of the composite wine vector (variety, region, price, points), driven by a structured user profile (e.g., "I like Pinot Noir, under $30, from France"). Represents the *status quo* / traditional approach a novice user would have to navigate (requires wine vocabulary).
3. **Model 2 — NLP-Embedding-Driven Content Filtering (core novel layer, RQ1/RQ2)**: the user's free-text description (e.g., "something fruity but not too sweet, easy to drink with dinner") is embedded with the same SBERT model and compared via cosine similarity directly to the `description` embeddings of all wines, optionally combined with soft structured filters (e.g., price ceiling). **This is the mechanism that directly answers the vocabulary-gap question** and is the centerpiece of the methodology chapter.
4. **Model 3 — NLP + Iterative Refinement (RQ3, headline novelty)**: after an initial Model 2 recommendation, the user provides a refinement ("less oaky, more like #2"). This is operationalized as a vector-arithmetic adjustment to the query embedding — e.g., using contrastive pairs mined from the corpus (wines described as "oaky" vs. "unoaked" within the same variety) to derive a "less oaky" direction vector, which is subtracted from the query embedding before re-ranking. A small (~30–50 query) labeled set, validated with the sommelier, is used to tune and sanity-check these direction vectors.
5. **Model 4 / Exploratory — Taster-Proxy Collaborative Filtering (RQ5)**: using `taster_name` and `points` as a (user, item, rating)-style matrix, apply a memory- or model-based CF technique (S12/S13) to explore whether reviewer-level "taste signatures" can be detected and combined with the Model 2/3 outputs (e.g., as a re-ranking signal). **Explicitly framed in the report as exploratory and small-n** (≈15–20 "users"), not as a production-ready personalization layer — this framing is itself part of the academic contribution (RQ5).

### 6.3 Explainability (lightweight, integrated rather than a separate "layer")
Rather than building a full Layer 5 XAI system, each recommendation in the demo is accompanied by a **template-based explanation** generated from the overlap between the query embedding's nearest semantic neighbors (top contributing words/phrases via embedding similarity to known descriptors) and the recommended wine's description — e.g., *"Recommended because its tasting notes emphasize ripe red fruit and soft tannins, matching your description."* This satisfies the spirit of Layer 5 (and is citable against the explainability literature, e.g., feature-based explanation styles in 1804.11192) without requiring a separate evaluation track.

### 6.4 Context-awareness (design-only, future work)
A short design section (informed by S18/S19) proposes how contextual pre-filtering (occasion, season, price tier) *would* be integrated — e.g., as a pre-filter on Model 2/3's candidate set — but is **not built or evaluated** in the prototype. This demonstrates command of the material without inflating scope.

---

## 7. Evaluation Framework

### 7.1 Offline Evaluation — Self-Retrieval Protocol (primary, addresses ground-truth problem)
Because no consumer interaction data exists, ground truth is constructed via a **held-out self-retrieval design**, standard in IR/embedding evaluation:
1. Hold out a random sample of wines (e.g., 200–300) and their professional descriptions.
2. For each held-out wine, generate a **casual paraphrase** of its tasting note in plain, non-technical language (a manual/structured rewriting task — *not* AI-generated verbatim text, to remain consistent with the AI policy; can be partially template-assisted but reviewed and adjusted by Paul, with methodology disclosed) — these paraphrases become the "user queries."
3. For each query, check whether the system retrieves the original held-out wine (and/or wines from the same variety/style cluster) within the top-K.
4. Compute **Precision@K, Recall@K, NDCG@K, Coverage, Diversity (intra-list embedding distance), and Serendipity** (recommendations outside the most-popular/obvious cluster for that variety) for Models 0–3.

### 7.2 Expert Validation — Sommelier Panel (secondary, addresses credibility)
- A curated subset (≈30 queries spanning easy → ambiguous phrasing) is reviewed by the sommelier, who rates each model's top-5 recommendations for relevance (e.g., 1–5 Likert) and provides qualitative commentary on *why* a recommendation does/doesn't fit.
- This produces an **expert-labeled relevance set** that can recompute Precision@K/NDCG@K on a smaller, higher-confidence sample — directly strengthening RQ2's evidentiary basis beyond the self-retrieval proxy.
- Sommelier feedback on Model 3 (refinement) specifically tests whether the "less oaky" / "more fruity" direction-vector adjustments behave the way an expert would expect.

### 7.3 Refinement-Specific Evaluation (RQ3)
- Compare Precision@K/NDCG@K of Model 3's *post-refinement* ranking against Model 2's *pre-refinement* ranking on the same queries, using the sommelier panel to judge whether the refined recommendations move in the *intended direction* (not just a different direction).

### 7.4 Taster-Proxy CF Evaluation (RQ5)
- Standard CF evaluation (e.g., RMSE/MAE on held-out taster ratings, or rank correlation between taster-implied preferences and Model 2/3 rankings) — reported with explicit small-sample caveats.

### 7.5 Optional User Study (if time permits, Phase 4 of the roadmap)
- 10–20 self-identified wine novices interact with the demo and complete a short survey (satisfaction, perceived relevance, ease of use, trust — as originally scoped in the Data Strategy document), with a basic informed-consent form. Framed as a **supplementary, exploratory** signal — not the primary evidentiary basis (so its omission, if time runs short, doesn't threaten the core RQ answers).

### 7.6 Statistical Treatment
- Paired comparisons (e.g., Wilcoxon signed-rank test) on per-query metric differences between Models 0–3, to support claims of improvement with appropriate statistical caveats — a detail that meaningfully differentiates a "Notable" from a "Sobresaliente" written report.

---

## 8. Report Structure

1. **Introduction** — problem framing (Gen Z/wine disengagement, L.E.K. 2026 evidence), business context (LVMH), RQ and sub-RQs, contribution statement, report roadmap.
2. **Literature Review** — three synthesized themes: (a) recommender-systems theory (content-based, collaborative, hybrid, CARS, cold-start, explainability — S06/S12/S15/S18/S19/S21/S22 plus the systematic CARS review and SBERT paper); (b) wine consumer behaviour & the vocabulary/self-confidence barrier (Label Fluency; Barber/Dodd/Ghiselli; Gen Z reports); (c) NLP/embedding-based personalization and the identified gap SensoryRec fills.
3. **Data & Exploratory Analysis** — dataset description, summarized EDA findings (the figures already produced), missingness handling, ethical/licensing notes on the Kaggle dataset.
4. **Methodology** — feature engineering pipeline; the Model 0–4 ladder with architecture diagrams; the natural-language preference-mapping mechanism (Model 2) and refinement mechanism (Model 3) in full technical detail; explainability template design; context-awareness design sketch (not built).
5. **Evaluation Design** — the self-retrieval protocol, sommelier panel design and instruments, optional user-study design, statistical methods — written so it could be replicated.
6. **Results** — the ablation table (Models 0–3) across Precision@K/Recall@K/NDCG@K/Coverage/Diversity/Serendipity; refinement-specific results (RQ3); sommelier panel findings; taster-proxy CF exploration (RQ5); user-study results if conducted.
7. **Discussion** — answer each sub-RQ explicitly; interpret where natural-language input helps/hurts relative to attribute-based filtering; honest discussion of limitations (dataset interaction-data gap, sommelier panel size, paraphrase-generation methodology, small-n CF proxy).
8. **Business Implications** — productization narrative for LVMH, link back to L.E.K. data, premiumization argument, demo walkthrough.
9. **Conclusion & Future Work** — summary of contributions; explicit roadmap for context-awareness (Layer 4), fuller explainability (Layer 5), real-interaction-data collaborative filtering, and the spirits extension — i.e., the original 5-layer vision is preserved *as a roadmap*, demonstrating ambition without scope risk.
10. **References** (APA, per AI-policy formatting guidance).
11. **Appendices** — GitHub repo link & structure, sommelier instrument, consent form (if user study conducted), additional tables/figures, full ablation results.

---

## 9. Prototype Scope

**In scope (built, evaluated, demoed):**
- Full data pipeline on the Wine Enthusiast dataset (129,971 reviews), feature engineering as specified.
- Models 0–3 (non-personalized baseline → attribute-based content filtering → NLP-embedding-driven filtering → NLP + iterative refinement), fully implemented and evaluated under the Section 7 protocol.
- Template-based explanation generation accompanying Model 2/3 recommendations.
- Taster-proxy CF exploration (Model 4 / RQ5), implemented at a basic level (e.g., simple matrix factorization or item-item CF on the taster×wine matrix) with explicit small-sample framing.
- A lightweight **demo web application** (e.g., Streamlit/Flask) where a user types a free-text description, optionally refines it ("less sweet"), and receives top-5 wine recommendations with explanations — this is the tangible artifact for the oral defense.
- Sommelier validation session(s) on a curated query set.
- Full GitHub repository: code, documentation, reproducibility instructions.

**Explicitly out of scope (future work, designed/discussed but not built):**
- A fully implemented context-aware (Layer 4) recommendation engine — addressed as a design proposal in the methodology chapter only.
- A full XAI framework beyond template-based explanations (Layer 5) — current approach is sufficient for the explainability sub-discussion without a dedicated evaluation track.
- Real-time integration with Wine-Searcher / geolocation purchasing layer — mentioned in business implications as a productization step, not built.
- The spirits extension — explicitly deferred to future work, with a short paragraph on how the same Model 2/3 architecture would generalize (and what would need to change, e.g., different sensory vocabulary corpus).
- The 10–20 person user study — included as an optional Phase 4 addition (per the roadmap) if time allows, but the project's core RQs do not depend on it.

---

## Why this scope is grade-optimized

- Every element of Sections 2–9 traces back to material **already present in the project workspace** — no new literature or data sources are required, only synthesis, specification, and disciplined scoping.
- The **Model 0–3 ablation ladder** gives the written report a clean, examiner-friendly spine: each model's incremental value is a direct, quotable result.
- The **self-retrieval + sommelier panel** evaluation design solves the "no ground truth" problem without fabricating synthetic users, directly addressing the validity risk flagged previously.
- The **taster-proxy CF exploration** preserves the "hybrid" framing and the course's collaborative-filtering content (S12/S13) honestly, at small scale, rather than either omitting CF entirely or relying on simulated users.
- **Context-awareness, full XAI, and spirits** remain visible as a coherent future-work roadmap — preserving the ambition and breadth of the original 5-layer vision (valuable for the oral defense's "vision" component) while keeping the *built and evaluated* core tight enough to execute with rigor in the available time.
