# Meeting 16 Nov 2026: what to say (for you, not the supervisor)

About 25 minutes. Open `1_progress_since_last_meeting.md` on GitHub and walk
through it top to bottom; this page is what to say at each section.

## Opening (one sentence)

"Everything you asked for last time is done and has a results file behind it.
I also have the first accuracy numbers, with one surprise I want your view on,
and a first prototype of the app."

## 1. Roger Voss (1 min)

- "You were right to question it. It's real: Wine Enthusiast gives each
  region to one critic. He does 84% of France and 99% of Portugal."
- Volunteer the limitation: "A text model could partly learn a critic's
  style rather than the wine. I'll state that as a threat to validity."

## 2. X-Wines (3 min)

- "Vivino forbids scraping, so I used X-Wines, a published dataset of 21
  million consumer ratings, licence CC0, from a 2023 journal paper."
- The key line: "Consumers are *more* concentrated than the critics: the top
  20% of wineries get 87% of all ratings, against 66% in our dataset. And the
  user × wine matrix is 0.02% full. So even with a million users, most wines
  have almost no ratings: that's the cold start my thesis is about."
- If asked "will you train on it?": "No, analysis only. It isn't linked to our
  wines, and collaborative filtering is outside the agreed scope."

## 3. The 1,000 simulated queries (5 min, the main ask)

- Explain the trick: "If you ask an LLM for 1,000 different requests it
  repeats itself, so I build 1,000 different wish-lists with a fixed seed and
  Gemini only phrases each one as a sentence." Read one example aloud.
- Show the table and the curve:
  - "Popularity gives the same 5 wines to everyone."
  - "The embedding models reach the most of the catalogue, and the new open
    model, bge-small, reaches the most: about 2,000 different wines."
  - "The curves are still rising: more users would still discover new wines."
- Say the caveats before he does:
  - "Coverage looks small, but 5,000 recommendations can cover at most ~4% of
    120,000 wines."
  - "This measures breadth, not whether the wines are right. Random
    recommendations would score well here, so I read it together with the
    accuracy test."
  - "Diversity within a list is lowest for Sentence-BERT, but that's partly
    built in: I measure diversity with Sentence-BERT's own embeddings, so its
    lists look coherent by construction."

## 4. Embeddings and Gemini (2 min)

- "I added an open model, bge-small, to every comparison."
- "Gemini writes the simulated queries. The model in the course notebook has
  been retired, and the free tier gives 20 requests a day on Flash, so I used
  Flash Lite at 500 a day. It's all in DECISIONS.md."
- "Gemini embeddings are next if the free quota allows it."

## 5. Maps (1 min)

- "I compared folium, plotly and pydeck. Our data has no coordinates, only
  place names; plotly can place a country from its code, so I chose it."
- "The mock-up shows the real countries of origin of the top 5, and a
  'where to buy' layer with invented shops, labelled MOCK on every pin.
  Maps stay Future Work, as agreed."

## 6. Accuracy (5 min)

- Say it straight, before he reads it: "The accuracy result is not what I
  expected. On the exact-wine test, TF-IDF finds the wine 19 times out of 200,
  bge-small 6 times, Sentence-BERT once."
- Then the explanation: "The LLM queries keep about a quarter of the critic's
  words, and rare copied words lead TF-IDF straight to the note. And one
  correct wine out of 120,000 is very strict: 'a smooth fruity red'
  describes thousands of wines. The semantic models give the right kind of
  wine, just rarely that exact one: for a red request they give five reds,
  TF-IDF gives a Chardonnay first."
- "So the exact-wine test alone can't tell if a recommendation is good. I'd
  like your view on adding graded relevance (sommelier ratings of the top 5)
  and a closeness measure, and the human queries will test the copying
  explanation."
- Refinement is the good news: "'Sweeter' makes the list sweeter 88% of the
  time, against 6% for a random direction of the same size."
- Don't defend Sentence-BERT. The honest finding is a result too, and the
  long-tail and coverage results still hold.

## 7. Prototype (3 min)

- It already uses the "French Maison" look (palette, fonts, category
  badges); say the full design is a proposal for S3.
- Run `make app` on your Mac (after `make data` and `make embed`), or show
  the two screenshots.
- Type one of the demo queries, choose `semantic`, then press "sweeter" or
  "cheaper".
- "The ranking is deterministic, the same models as in the evaluation. The
  LLM will only write the explanation, from the retrieved notes; that's the
  next sprint."

## 8. Corrections (1 min)

Go through the four points calmly. Main one: "The top 20% of wineries hold
66% of reviews, not 30% as I said."

## Questions to ask him

1. The accuracy surprise: is graded relevance (sommelier ratings) plus a
   closeness measure the right way to complete the exact-wine test?
2. Is the simulated-query study, read together with the accuracy test,
   enough to answer the long-tail question, or does he want more queries?
3. Is X-Wines enough as "user data", given it is analysis only?
4. For S3, Streamlit or Gradio? (The prototype uses Streamlit.)
5. Gemini embeddings: worth the effort, or is one open model enough?
6. Ethics for the ~30 human queries: whom should I ask at IE?
7. Design: I want the final demo to look like a premium product ("French
   Maison" brief; the prototype already uses its palette and fonts). Can the
   scope include a **wine-only** version of it: homepage, Find Your Bottle,
   results with explanations, product page? Spirits, "My Cellar" and
   purchasing stay out unless he says otherwise.
8. Would an optional step-by-step questionnaire that *writes* the free-text
   request for the user be acceptable, or does it weaken the "own words"
   research question?

## Be ready for

| Question | Answer |
|---|---|
| "Did the LLM pick the wines?" | No. It only wrote the customers' sentences, and the test queries. Ranking is cosine similarity or TF-IDF scores. |
| "Why are the simulated queries so long?" | Each wish-list can hold up to six wishes. Real novices would be shorter: a stated limitation. |
| "Why should I trust Gemini's phrasing?" | Every query passed the 85-word expert check, all answers are saved, and the ~30 human queries will check realism. |
| "Who wrote the code?" | "I built it with Claude Code, every commit is marked. I made the methodological decisions, and they're dated in DECISIONS.md." |
