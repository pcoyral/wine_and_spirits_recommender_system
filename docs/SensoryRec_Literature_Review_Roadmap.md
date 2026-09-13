# SensoryRec — Literature Review Roadmap

This roadmap organizes every academic/industry source currently in the project workspace into themes, classifies each as **foundational** or **optional**, identifies **missing papers** that would strengthen the review, extracts the **key theories** each theme contributes, and states the **gap in current research** that SensoryRec occupies. It closes with a proposed **conceptual framework** for the literature review chapter.

A note on source types: the "S-series" files (S02–S22) are **course materials**, not primary literature — they are extremely useful for identifying *which* theories matter and *which* primary sources to track down, but a thesis literature review should cite primary sources (papers, books, reports) wherever possible. Each theme below distinguishes course scaffolding from citable primary sources.

---

## Theme 1 — Recommender Systems Foundations: Content-Based, Collaborative, and Hybrid Approaches

**Foundational (course scaffolding — use to identify primary sources):**
- **S02 — Introduction to Recommendation Systems**: positions RS within the ML taxonomy (alongside supervised/unsupervised/RL); establishes why RS matter for businesses with large catalogs.
- **S06 — Approaches to Building Recommenders (Collaborative, Content-Based, Hybrid)**: the master reference for the CB-vs-CF tradeoff table (cold-start, over-specialization/filter bubble, data sparsity, domain-knowledge dependency, scalability, transparency) and hybrid combination strategies.
- **S12 / S12 (Answers) — Theory of Collaborative Filtering**: memory-based (user-based/item-based) vs. model-based (SVD) CF; the "shared latent space for users and items" concept.
- **S15 — Theory of Content-Based Recommenders** and **S16 — Building / Integrating Content-Based Recommenders into Chatbots**: the direct conceptual basis for Models 0–2.
- **S09 — Building a Non-Personalized Recommender**: the basis for Model 0 (baseline).

**Optional (cited within S06/S12/S15 — track down for the actual review):**
- *"A Comprehensive Survey of Neighborhood-Based Recommendation Methods"* (Xia Ning et al.) — referenced repeatedly across S06/S12/S15 as "further reading" for CF/CB pros and cons; a strong, citable primary source for the foundational CB/CF comparison.

**Key theories:**
- Cold-start, over-specialization (filter bubbles), data sparsity, domain-knowledge dependency, scalability, transparency as the canonical CB-vs-CF tradeoff axes.
- Memory-based vs. model-based CF; shared latent space for users and items.
- Hybridization strategies (weighted, switching, mixed, feature-combination, cascade, feature-augmentation, meta-level) — S06's hybrid diagram explicitly shows "combine collaborative filtering learned on feedback data with content-based learned on item metadata."

**Contribution to SensoryRec:** This theme supplies the **vocabulary and justification for the Model 0–3 ablation ladder**. SensoryRec's Models 2–3 are best classified as a **feature-augmentation hybrid** (a content-based system augmented with an NLP-derived similarity signal) — naming this explicitly, with citation to the hybrid taxonomy, demonstrates command of the theory rather than just building "a recommender that uses embeddings."

---

## Theme 2 — Cold-Start, Bias, and the Explore-Exploit Tradeoff

**Foundational:**
- **S21 — Addressing the Cold-Start and Bias Problems**: directly frames SensoryRec's target population (users with no interaction history) as a textbook cold-start scenario, and covers selection bias (missing-not-at-random ratings) — directly relevant to honestly discussing why the Wine Enthusiast dataset cannot support standard CF.

**Optional:**
- **S22 — The Explore-Exploit Tradeoff and Multi-Armed Bandits** (theory + two Colab notebooks: practical exercise and "bandits with real-world data"): not core to the current scope, but directly relevant to **future work** — Model 3's iterative refinement could be reframed as a bandit-style exploration problem in a follow-up system.

**Key theories:**
- Cold-start typology (new user / new item / new system).
- Selection bias and the "missing not at random" assumption (Marlin et al., cited in S21) — critical for honestly framing the Wine Enthusiast `points`/`taster_name` data as non-random.
- Exploration vs. exploitation; epsilon-greedy and Bayesian (Beta-distribution) bandit updating (S22).

**Contribution to SensoryRec:** S21 is the theoretical anchor for the **entire problem framing** — SensoryRec is, fundamentally, a cold-start solution. This theme should be cited early in the literature review (Section 2.3 in the proposed structure below) to establish *why* natural language is being proposed as the mechanism, before NLP/embeddings are introduced as the *how*.

**Missing paper:** Lika, Kolomvatsos & Hadjiefthymiades (2014), *"Facing the Cold Start Problem in Recommender Systems"* (Expert Systems with Applications) — a widely-cited cold-start survey that categorizes mitigation strategies (demographic, content-based, hybrid, active learning) and would let SensoryRec position "natural-language elicitation" as a distinct category alongside the established ones.

---

## Theme 3 — Context-Aware Recommender Systems (CARS)

**Foundational:**
- **ContextAware_Recommender_Systems.pdf** (Adomavicius & Tuzhilin, book chapter) — **the** foundational CARS reference. Defines the `<user, item, context, rating>` data model and the three paradigms: contextual pre-filtering, post-filtering, and contextual modeling, plus combining multiple pre-filters via an ensemble/aggregator.
- **2007.04782v1** — Santana & Domingues (2020), *"A Systematic Review on Context-Aware Recommender Systems using Deep Learning and Embeddings"* — a systematic review (16 primary studies) showing that **embeddings (especially Word2Vec-style) are already a established mechanism for incorporating context into RS**, often combined with RNNs for sequential/session data.

**Optional (course scaffolding):**
- **S18 — Theory of Context-Aware Recommenders** and **S19 — Building a Context-Aware Recommender**: practical grounding for the design-only Layer 4 discussion.

**Key theories:**
- Pre-filtering (context as a query filter on the data before recommending), post-filtering (context adjusts/reranks output of a traditional RS), and contextual modeling (context as an explicit predictor, `Rating = R(User, Item, Context)`).
- Combining multiple contextual pre-filters via an ensemble/aggregator, used only where they outperform the non-contextual baseline.
- Embeddings (Word2Vec/item2vec-style) as a representation that captures contextual/sequential information for downstream DL models (2007.04782).

**Contribution to SensoryRec:** Provides the citable basis for the **future-work design sketch** of Layer 4 (e.g., "occasion" or "price tier" as a pre-filter on Model 2/3's candidate set, per Adomavicius & Tuzhilin's pre-filtering paradigm). 2007.04782 is doubly useful: it also belongs to Theme 4 (NLP/Embeddings), and explicitly bridges CARS and embeddings — useful for arguing that SensoryRec's embedding-centric architecture is *extensible* toward full CARS.

**Gap worth naming explicitly:** Table 3 of 2007.04782 lists the domains where embeddings+DL CARS have been applied (citation recommendation, music, POI, e-commerce, fashion, news, video, academic collaboration) — **none involve a catalog with rich descriptive text but no interaction data**, which is exactly the Wine Enthusiast/wine-recommendation situation. This absence is part of SensoryRec's positioning.

---

## Theme 4 — Natural Language Processing and Embeddings for Personalization (Core Technical Theme)

**Foundational:**
- **D191410.pdf** — Reimers & Gurevych (2019), *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"* (EMNLP-IJCNLP 2019). **This is the single most load-bearing technical citation in the thesis** — it is the method (via `all-MiniLM-L6-v2`) underlying every wine-description embedding and every user-query embedding in Models 2–3. The siamese/triplet network architecture and the use of cosine similarity over sentence embeddings for semantic similarity search are the direct technical justification for "embed the user's free-text query and the wine's tasting note in the same space and compare."
- **2007.04782v1** (also Theme 3) — establishes that Word2Vec-style embeddings are the dominant technique for representing items/context in modern RS, situating SBERT-based wine embeddings within an established (if not wine-specific) tradition.

**Optional (course scaffolding):**
- **S16 — Integrating Content-Based Recommenders into Chatbots**: practical scaffolding for the natural-language *interface* (how a user's free text gets turned into a query against the recommender) — useful for the Implementation chapter, less so for the literature review itself.

**Key theories:**
- Siamese/triplet network training for semantically meaningful sentence embeddings (Reimers & Gurevych).
- Semantic Textual Similarity (STS) as the evaluation paradigm for embedding quality — directly informs how Model 2's "query ↔ description" matching should be validated.
- Embeddings as a general-purpose representation for recommendation context/items (2007.04782).

**Contribution to SensoryRec:** This is the theme that makes **Model 2 defensible** as more than "we called an SBERT API" — it grounds the choice of model, the similarity metric, and the evaluation paradigm (STS-style) in a specific, well-cited methodology.

**Missing papers (high priority — these are the most important gaps in the entire corpus):**

1. **Mikolov, Chen, Corrado & Dean (2013), *"Efficient Estimation of Word Representations in Vector Space"* (Word2Vec)** — referenced within 2007.04782's bibliography but not itself present in the project workspace. This is the **foundational paper for embedding-space vector arithmetic** (the famous "king − man + woman ≈ queen" analogy property), and is the **direct theoretical justification for Model 3's "less oaky" direction-vector mechanism**. Given Model 3 is the headline novel contribution (RQ3), this paper should be added and cited explicitly.

2. **Critiquing-based / conversational recommender systems** — e.g., Chen & Pu's body of work on critiquing in recommender interfaces, or Christakopoulou, Radlinski & Hofmann (2016), *"Towards Conversational Recommender Systems"* (KDD 2016). This is the **academic home for "more X, less Y" iterative feedback** — exactly what Model 3 implements. Its absence from the current corpus is the single biggest literature gap relative to the project's stated novelty (RQ3). Without it, Model 3 risks looking like an engineering trick rather than a contribution situated in an existing research tradition (which it is — critiquing-based RS is a 20+ year old subfield).

3. **Prior NLP/embedding-based recommenders in food, beverage, or wine domains** — a targeted search (Google Scholar / Semantic Scholar) for "wine recommender system NLP," "wine embeddings recommendation," or papers using the Wine Enthusiast / `winemag-data-130k` dataset for recommendation (rather than just sentiment/rating prediction, which is the more common use of this dataset) is needed. If such papers exist, they become the **closest prior art** and must be addressed directly. If a systematic search finds none, that absence itself becomes part of the gap statement (Section "Gaps," below) — but this needs to be *demonstrated*, not assumed.

---

## Theme 5 — Explainable Recommendation (XAI for Recommender Systems)

**Foundational:**
- **1804.11192v7** — Zhang & Chen (2019/2021), *"Explainable Recommendation: A Survey and New Perspectives."* Introduces the **5W framework** (what, when, who, where, why) for positioning explainability within RS research, and a **two-dimensional taxonomy**: information source/display style (relevant user/item, feature-based, textual sentence, visual, social) crossed with algorithmic mechanism.
- **2202.06466v1** — Chen, Zhang & Wen (2022), *"Measuring 'Why' in Recommender Systems: A Comprehensive Survey on the Evaluation of Explainable Recommendation."* Reviews 100+ papers on **how to evaluate explanations** (user-centric: trust, satisfaction, persuasiveness, scrutability vs. system-centric: fidelity).

**Optional:**
- **2305.11755v3** — (Chatti et al.), *"Visualization for Recommendation Explainability: A Survey and New Perspectives."* Focuses specifically on **chart/visualization-based** explanations. SensoryRec's planned explanations are textual/template-based (per the scoped methodology), so this is **optional** unless the report adds a "future work: visual explanation dashboard" subsection — in which case this becomes directly relevant.

**Key theories:**
- The 5W framework for situating *why* explainability matters (transparency, persuasiveness, effectiveness, trustworthiness, satisfaction, debugging).
- Feature-based explanation style (Zhang & Chen) — explanations grounded in matching item attributes/features to user-profile features — is the **closest taxonomy match** for SensoryRec's planned template explanations ("recommended because its tasting notes emphasize ripe red fruit...").
- User-centric vs. system-centric evaluation of explanations (Chen, Zhang & Wen) — directly informs whether/how to extend the sommelier panel to rate explanation quality, not just recommendation relevance.

**Contribution to SensoryRec:** Provides the citation backbone for the lightweight explanation layer described in the scoped methodology, and a ready-made evaluation lens (2202.06466) if time allows the sommelier panel or user study to assess explanations specifically.

**Gap worth naming:** Most feature-based explanation methods surveyed in 1804.11192 assume access to either (a) interaction-based latent factors or (b) structured knowledge graphs. **SensoryRec's explanations are derived purely from embedding-space similarity between free-text and tasting notes — a content-only, interaction-free explanation source** that is not a prominent case in the surveyed taxonomy. This is a small but genuine positioning point.

---

## Theme 6 — Recommender System Evaluation: Accuracy and Beyond

**Foundational (course scaffolding):**
- **S07 — Performance Metrics for Recommendation Engines**: Precision@K, Recall@K, NDCG@K, and beyond-accuracy metrics (Coverage, Diversity, Serendipity, Novelty).
- **S10 — Data Splitting and Evaluation Methods (Full Version)**: train/test/validation splitting strategies appropriate for RS (temporal split, leave-one-out, k-fold) — directly informs how the self-retrieval holdout (Section 7.1 of the methodology) should be constructed and justified.

**Optional:**
- **S08 — Practical Exercises on Performance Metrics** and the shorter **S10** Colab: implementation-level scaffolding, useful for the code but not for citation.
- **Recommender_Systems_Not_Just_Recommender_Models_.pdf** (NVIDIA Merlin / Medium, Even Oldridge) — a practitioner perspective arguing that production RS quality depends on the *whole system* (data pipelines, serving, monitoring), not just model accuracy. **Optional** as an academic citation (it's a blog post, not peer-reviewed), but useful for framing the Business Implications/demo chapter's productization narrative — cite cautiously, perhaps as an industry-perspective footnote rather than a core source.
- **2202.06466** (also Theme 5) — overlaps here specifically for evaluating *explanations*.

**Key theories:**
- Rank-aware accuracy metrics (Precision@K, Recall@K, NDCG@K) vs. beyond-accuracy metrics (Coverage, Diversity, Serendipity, Novelty) — the latter directly answer RQ4.
- Appropriate train/test splitting for RS (avoiding temporal leakage, leave-one-out for sparse data).
- The "systems, not just models" framing for evaluating deployed recommenders.

**Contribution to SensoryRec:** This theme is the **direct methodological basis for Section 7's evaluation framework** — every metric named in the Data Strategy document (Precision@K, Recall@K, NDCG@K, Coverage, Diversity, Serendipity) traces back to S07.

**Missing paper:** A classic or foundational reference for **evaluation without click/interaction logs** — e.g., the Cranfield paradigm from information retrieval (Cleverdon, 1967) or, more practically, the **STS benchmark methodology referenced within Reimers & Gurevych (2019) itself** (worth checking D191410's own references — it likely cites the STS Benchmark / SemEval tasks, which use human-judged similarity scores as ground truth in a structure directly analogous to the proposed self-retrieval + sommelier protocol). Tracing this citation chain would give Section 7.1's design a much stronger methodological pedigree than "we made this up because there's no interaction data."

---

## Theme 7 — Wine Consumer Behaviour: Vocabulary, Confidence, and Knowledge

**Foundational:**
- **Label_Fluency_and_Consumer_SelfConfidence.pdf** — Barber, Ismail & Taylor (2007), *Journal of Wine Research*. Establishes that **self-confidence is distinct from wine expertise/involvement**, and that consumers with low self-confidence prefer simpler, more familiar information presentation. This paper is the **closest existing academic articulation of the "vocabulary gap"** that SensoryRec addresses — though it is framed around label design, not recommendation systems.
- **BarberN_DoddT_H_GhiselliR_2008_Capturingtheyoungerwineconsumer.pdf** — Barber, Dodd & Ghiselli (2008). Generational segmentation (Millennial vs. Generation X), documenting that under-30 wine consumption has historically been flat (6% of wine consumed by 18–24 year-olds per Wine Market Council 2006) while Generation X favors beer over wine. Establishes the **historical baseline** against which the 2026 L.E.K. data (Theme 8) shows a potential shift.

**Optional:**
- **Wine_consumers_environmental_knowledge_and_attitu.pdf** — Barber, Taylor & Strick (2009), *International Journal of Wine Research*. Finds that **objective** environmental knowledge predicts attitude more strongly than **subjective** (self-assessed) knowledge, which then predicts willingness to purchase. Tangential to the core RQ, but (a) the subjective-vs-objective knowledge distinction is conceptually parallel to the "vocabulary gap" construct (consumers may *feel* unknowledgeable about wine even when their actual taste preferences are coherent and expressible in plain language), and (b) could support a brief "future extension: sustainability-aware recommendation" paragraph in Future Work.

**Key theories:**
- Self-confidence as a construct distinct from product expertise/involvement (Barber et al., 2007) — directly supports framing SensoryRec's target user as "has preferences, lacks confidence to express them in wine vocabulary," not "lacks preferences."
- Processing/label fluency theory (cited within Barber et al. 2007 via Novemsky et al.) — "a target is evaluated more positively the more easily it can be evaluated" — provides a psychological mechanism for *why* lowering the vocabulary barrier should improve engagement, not just recommendation accuracy.
- Subjective vs. objective knowledge as differentially predictive constructs (Barber, Taylor & Strick, 2009).
- Generational cohort theory: Millennials as internet-native, value-driven, "fun + responsibility" (Barber, Dodd & Ghiselli, 2008).

**Contribution to SensoryRec:** This theme is the **psychological/behavioural anchor for the entire problem statement**. It should be introduced early (Section 2.2 or 2.3) to establish *why* "vocabulary" is the right lever to pull, before the technical themes explain *how*.

**Important framing note for the gap statement:** These three papers are **2007–2009 vintage**. They predate smartphone-era wine-discovery apps (Vivino launched 2010), SBERT-era NLP (2019), and the 2026 Gen Z resurgence data (Theme 8). The literature review must **explicitly bridge this ~17-year gap** — e.g., "the self-confidence/vocabulary barrier identified by Barber et al. (2007) has not, to our knowledge, been addressed via natural-language-driven recommendation technology that did not exist at the time."

**Missing paper:** A more recent (2015+) consumer-behaviour or HCI study on **digital wine discovery tools** (e.g., studies of Vivino, wine-matching apps, or sommelier-chatbot interfaces) would substantially strengthen this bridge and should be sought.

---

## Theme 8 — Gen Z, Wine, and the Market Context (Business/Industry Literature)

**Foundational:**
- **winningthefuturewineconsumer.pdf** — L.E.K. Consulting (2026 survey, N=110–747 depending on segment). The most rigorous, most recent, and most directly load-bearing source for the business case: Gen Z wine participation rising (60% report higher consumption over the past two years), structural headwinds (health perception shifts, GLP-1 medications, RTD/seltzer competition), premiumization (bottles ≥$15 showing relative resilience, ~65–73% share over 2019–2025), and — critically — **"improving my ability to find wines I enjoy" is the #1 stated reason nonconsumers would start drinking wine, ahead of social occasions, price, and familiarity.**

**Optional:**
- **Declining_Alcohol_Consumption_of_Gen_Zers.pdf** ("Sip Happens: Why Gen Z is Toasting Less," Quals.AI/Twentify, 100 in-depth interviews, four US states). Qualitative and US-centric; useful for color/quotes on Gen Z attitudes but methodologically lighter than L.E.K.
- **Cheers_to_Change__How_GenZ_is_Redefining_Drinking_Culture.pdf** — Steedley & Thomas (2024), Georgia Southern University, *Eagles Talking About the Public's Health*. A public-health commentary (not peer-reviewed original research) on *general alcohol* (not wine-specific) decline among Gen Z — useful for the broader "why is this happening" context (health-consciousness, "hustle culture," prolonged adolescence) but should be cited as commentary, not as primary research.

**Key theories/findings:**
- Gen Z wine participation trajectory diverges by age sub-cohort (21–24 vs. 25–29) due to COVID-era formative experiences (L.E.K.).
- Premiumization trend — directly relevant to LVMH's portfolio positioning.
- Health-consciousness, "hustle culture," and prolonged adolescence as drivers of the broader Gen Z moderation trend (Cheers to Change) — useful context but requires care not to overstate causal claims given the source type.

**Contribution to SensoryRec:** This theme **motivates the Introduction and Business Implications chapters**. The L.E.K. finding on "ability to find wines I enjoy" is arguably the single most important citation in the entire thesis for establishing real-world relevance — it should be foregrounded in the opening paragraphs.

**Important framing note:** These are **industry reports and a public-health commentary, not peer-reviewed academic studies**. In the literature review, they should be clearly labeled as such (e.g., "industry survey evidence" vs. "academic literature") — mixing source types without distinction is a common quality issue external readers flag. The L.E.K. report's methodology (survey, stated Ns) should be described so its evidentiary weight is transparent.

---

## Consolidated List of Missing Papers (Priority Order)

1. **Mikolov et al. (2013), Word2Vec** — direct theoretical basis for Model 3's embedding-arithmetic refinement mechanism. *High priority.*
2. **Critiquing-based / conversational recommender systems** (e.g., Chen & Pu; Christakopoulou, Radlinski & Hofmann, 2016) — academic home for "more X, less Y" feedback; directly addresses RQ3. *Highest priority — this is the most significant gap relative to the project's stated novelty.*
3. **Prior NLP/embedding-based wine or food/beverage recommenders** — needed to establish (or rule out) closest prior art. *High priority — requires an active search, not just a library check.*
4. **Lika, Kolomvatsos & Hadjiefthymiades (2014), cold-start survey** — strengthens Theme 2's framing of natural language as a cold-start mitigation category. *Medium priority.*
5. **STS Benchmark / Cranfield-paradigm evaluation methodology** (likely traceable via D191410's own references) — strengthens the methodological pedigree of the self-retrieval evaluation protocol. *Medium priority.*
6. **Recent (2015+) digital wine-discovery consumer research** (Vivino-type studies) — bridges the 2007–2009 consumer-behaviour literature to 2026 application. *Medium priority.*
7. **Cross-domain natural-language preference-matching examples** (e.g., mood-based music recommendation, fashion recommendation from free-text descriptions — several such works appear only in the *reference lists* of 2202.06466/2305.11755, e.g., Andjelkovic et al.'s "Moodplay") — useful analogues showing the pattern generalizes beyond wine. *Lower priority, but easy wins since they're one citation-hop away.*

---

## Gaps in Current Research — Synthesis Statement

Taken together, the corpus supports the following gap statement:

> Recommender-systems research has established that **embeddings are an effective, increasingly common mechanism for representing items and context** (Theme 3/4), and that **content-based and hybrid approaches are well-suited to cold-start scenarios** (Themes 1/2). Separately, consumer-behaviour research on wine has established that **novice consumers are constrained by a vocabulary/self-confidence barrier rather than a lack of underlying preference** (Theme 7), and recent industry data confirms this barrier is currently the **single largest stated obstacle** to renewed Gen Z wine engagement (Theme 8). However, **no identified work connects these threads**: no system has been found that (a) takes free-form, non-technical natural-language descriptions of taste preference as direct input, (b) maps them into a semantic embedding space derived from professional reviews to solve the cold-start problem, and (c) is evaluated using methods appropriate for a content-rich, interaction-free domain. The closest adjacent literatures — critiquing-based/conversational RS (currently *missing* from the corpus, see above) and CARS+embeddings systematic reviews (Theme 3, which surveys no content-rich/interaction-free domains) — do not cover this specific configuration. **This is the gap SensoryRec occupies.**

This statement should only be finalized *after* the missing-paper searches above are conducted — particularly items 2 and 3, since either could reveal directly competing prior work that needs to be addressed rather than assumed absent.

---

## Proposed Conceptual Framework for the Literature Review Chapter

### Framework: "From Algorithmic Capability to Consumer Accessibility — A Convergence Model"

The literature review is organized as **three converging streams** that meet at SensoryRec's contribution:

- **Stream A — Algorithmic Capability** (Themes 1, 2, 3, 4, 5): what is computationally possible. Establishes that hybrid, embedding-based, context-aware, and explainable recommendation are mature techniques individually, but typically applied in interaction-rich domains.
- **Stream B — Consumer Need** (Theme 7): what the target user actually requires. Establishes that the barrier to entry is psychological/linguistic (confidence/vocabulary), not a lack of preference.
- **Stream C — Market Timing** (Theme 8): why this matters *now*. Establishes the 2026 Gen Z context and quantifies the vocabulary barrier as the top-ranked obstacle.
- **Cross-cutting — Evaluation Methodology** (Theme 6): how success will be measured, given the absence of interaction data — this stream doesn't "converge" so much as it threads through all the others, since every claim made via Streams A–C needs to be evaluated under the same protocol.

### Proposed Chapter Structure

| Section | Content | Themes | Closing element |
|---|---|---|---|
| 2.1 | Introduction & chapter roadmap | — | — |
| 2.2 | Foundations of Recommender Systems: CB, CF, and Hybrid Approaches | 1 | "Implications for SensoryRec" |
| 2.3 | The Cold-Start Problem: Why Novice Users Are Different | 2 | "Implications for SensoryRec" |
| 2.4 | Context-Aware Recommendation and the Rise of Embeddings | 3 | "Implications for SensoryRec" |
| 2.5 | Natural Language Processing for Personalization (core section — longest) | 4 | "Implications for SensoryRec" |
| 2.6 | Explainability in Recommender Systems | 5 | "Implications for SensoryRec" |
| 2.7 | Evaluating Recommenders Without Interaction Data | 6 | "Implications for SensoryRec" |
| 2.8 | The Wine Consumer: From the Vocabulary Barrier to the Gen Z Opportunity | 7 + 8 | "Implications for SensoryRec" |
| 2.9 | Synthesis: Positioning SensoryRec at the Convergence | All | Gap statement + framework diagram |

**Design notes:**
- Each section (2.2–2.8) ends with a short **"Implications for SensoryRec"** paragraph — this mirrors the style already used successfully in the project's own EDA Report and is exactly the kind of "synthesis, not summary" that distinguishes a strong literature review from an annotated bibliography.
- Section 2.5 (NLP for Personalization) should be the **longest and most detailed**, since it covers the core technical mechanism (Models 2–3) and is where the missing critiquing-based RS and Word2Vec citations (once added) carry the most weight.
- Section 2.9 is where the **convergence diagram** belongs — a simple three-stream diagram (Algorithmic Capability / Consumer Need / Market Timing converging on "SensoryRec") gives examiners a single visual that summarizes the entire review's argument. This is a natural candidate for a figure in the written report — happy to produce a draft diagram if useful.
- Throughout, **clearly distinguish source types** in-text (peer-reviewed academic paper vs. industry survey vs. course material/secondary source) — this is a small habit that meaningfully signals academic maturity to external readers, particularly given Theme 8's reliance on industry reports.

---

## How to Use This Roadmap

1. **Before the Oct 15 proposal**, the ≥5 required academic sources should be drawn primarily from Themes 4 (D191410), 3 (ContextAware_Recommender_Systems, 2007.04782), 7 (Label Fluency, Barber/Dodd/Ghiselli), and 8 (L.E.K.) — these five alone already tell the core story (technical mechanism + consumer problem + market motivation).
2. **Between now and the Jan 25 literature review deadline**, prioritize closing the missing-paper gaps in this order: (2) critiquing-based/conversational RS, (3) prior wine/F&B NLP recommenders, (1) Word2Vec, (5) STS/evaluation methodology, (4) cold-start survey, (6) digital wine consumer research, (7) cross-domain analogues.
3. Use the **"Implications for SensoryRec"** discipline from the start — for every paper read, write that paragraph immediately, even in rough form. This is what turns the literature review from a list into a synthesis, and it's the single highest-leverage writing habit for this section of the thesis.
