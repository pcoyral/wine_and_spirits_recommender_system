# Storyboard: the worked example (the "mínimo un caso")

> DRAFT for one slide. This is a mock-up of the *planned* demo (S3 chat and
> S4 images). The wines are placeholders, and no real recommendations appear
> until the system produces them.

## The scenario

A wine novice describes what she likes in her own words and gets five bottles
she can recognise in a shop. She refines the list once in plain language, then
reads why each bottle fits, with every explanation drawn only from the
critics' notes.

## Six frames

| # | Frame | What the user sees | What the system does | Built in |
|---|---|---|---|---|
| 1 | Persona | Lucía, 24, likes wine but knows no wine vocabulary. She is buying a bottle for a dinner with friends | | |
| 2 | Query | She types *"something red that isn't too heavy, a bit fruity, easy to drink"* | Sentence-BERT turns the sentence into a vector and compares it with about 120,000 tasting notes | S1 (done) |
| 3 | Top 5 | Five cards, each with a bottle photo, name, country, price and one plain-language line | Ranking by cosine similarity (deterministic, no LLM) | Ranking S1 (done); cards S3; photos S4 |
| 4 | Refinement | She types *"a bit lighter, and cheaper"* | "Lighter" moves the query away from the "body" direction, and "cheaper" caps the price at the median of the current top 5 | S2 (code done); chat S3 |
| 5 | Explanation | 2 or 3 sentences per bottle, such as *"The critic describes it as …"*, with the quoted words highlighted | The LLM writes each explanation only from the retrieved notes. A sample of 50 is checked by hand for unsupported claims | S3 |
| 6 | In the shop | The bottle photo and the label text to look for: producer, wine name, vintage | The photo comes from the curated subset. Other wines get a generic red, white, rosé or sparkling illustration plus the label text | S4 |

## Where the bottle images appear

- Frames 3 and 6. Only the 50 to 100 wines in the demo subset get a real photo,
  and each photo's source and licence are logged in `data/images/sources.csv`.
- Every other wine gets a generic illustration by colour plus the label text,
  so the app never shows the wrong bottle.

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

- The LLM never chooses the wines. It only explains a list that a
  deterministic model has already ranked, which keeps the results reproducible.
- The images help usability and are not part of any research claim. Coverage
  will be reported as it is ("photos for N of 119,954 wines").
