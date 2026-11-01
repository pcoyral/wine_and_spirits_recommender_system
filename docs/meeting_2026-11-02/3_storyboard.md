# Storyboard — the worked example (the "mínimo un caso")

> **DRAFT for one slide.** This is a mock-up of the *planned* demo (S3 chat +
> S4 images). The wines are placeholders: no real recommendations are shown
> until the system produces them. Adjust the persona and wording to your taste.

## The scenario in one line

A wine novice describes what she likes in her own words, gets five bottles she
can recognise in a shop, refines once in plain language, and reads why each
bottle fits — explained only from the critics' notes.

## Six frames

| # | Frame | What the user sees | What the system does | Built in |
|---|---|---|---|---|
| 1 | **Persona** | Lucía, 24, likes wine but knows no wine words; buying a bottle for a dinner with friends | — | — |
| 2 | **Query** | Types: *"something red that isn't too heavy, a bit fruity, easy to drink"* | Sentence-BERT turns the sentence into a vector and compares it with ~120,000 tasting notes | S1 ✅ |
| 3 | **Top 5** | Five cards: **bottle photo**, name, country, price, one plain-language line | Ranking by cosine similarity (deterministic, no LLM) | Ranking S1 ✅ · cards S3 · photos S4 |
| 4 | **Refinement** | Types: *"a bit lighter, and cheaper"* | "Lighter" moves the query away from the "body" direction; "cheaper" caps the price at the median of the current top 5 | S2 ✅ (code) · chat S3 |
| 5 | **Explanation** | 2–3 sentences per bottle: *"The critic describes it as …"* with the quoted words highlighted | The LLM writes the explanation **only from the retrieved notes**; a sample of 50 is checked by hand for unsupported claims | S3 |
| 6 | **In the shop** | The bottle photo and the label text to look for: producer, wine name, vintage | Photo from the curated subset; otherwise a generic red/white/rosé/sparkling illustration + the label text | S4 |

## Where the bottle images appear

- **Frame 3 and frame 6.** A real photo only for the ~50–100 wines of the demo
  subset, each with its source and licence logged (`data/images/sources.csv`).
- **Every other wine** gets a generic illustration by colour plus the label
  text — never a wrong bottle.

## Slide layout (sketch)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  SensoryRec — worked example (mock-up)                                   │
│                                                                          │
│ ① Lucía, 24, novice   ② "something red that isn't too heavy,            │
│    [persona icon]         a bit fruity, easy to drink"   [chat bubble]   │
│                                                                          │
│ ③ Top 5   [bottle][bottle][bottle][bottle][bottle]                       │
│           name · country · price · one plain line under each             │
│                                                                          │
│ ④ "a bit lighter, and cheaper"  →  ③ updates   (refinement)              │
│                                                                          │
│ ⑤ "The critic calls it 'bright and juicy' …"  (from the notes only)     │
│                                                                          │
│ ⑥ In the shop: [bottle photo] + label text to look for                   │
│                                                                          │
│  Ranking = Sentence-BERT (no LLM) · Explanation = LLM, grounded in notes │
└──────────────────────────────────────────────────────────────────────────┘
```

## Two things to say while showing it

- **The LLM never chooses the wines.** It only explains a list that a
  deterministic model already ranked, which keeps the results reproducible.
- **The images are a usability feature, not a research claim.** Coverage will
  be reported honestly ("photos for N of 119,954 wines").
