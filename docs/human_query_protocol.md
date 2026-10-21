# Protocol: hand-written casual queries

*Validity check for the LLM-written test queries (DECISIONS.md D1, D6).
About 30 people-written queries on 30 test wines are compared with the LLM
queries for the same wines (`results/llm_vs_human.csv`).*

> **Before recruiting anyone:** check IE University's rules for research
> with human participants (ethics approval or exemption, consent wording,
> data protection) with your supervisor or IE's research ethics contact.
> This protocol is a draft to show them; do not start until they confirm.

## What participants do

They read a professional tasting note once, the note is taken away, and
they describe the wine's taste in their own words, as they would to a
friend. Nobody drinks anything.

## Who takes part

- Adults who **know little about wine**: self-rating 1 or 2 on the scale below.
- Not wine professionals, wine students or regular wine-course attendees.
- Each participant writes for **3–5 cards**; aim for 6–10 participants so
  that every card is used exactly once.

Self-rated wine knowledge (asked before the first card):
1 = I know almost nothing about wine · 2 = a little · 3 = some · 4 = quite a lot · 5 = expert.

## Materials

- `data/human_cards.csv` (made by `make human-cards`): 30 cards, each with a
  card number and a tasting note. Print one note per card (large font, card
  number on the back), or put each note on its own slide.
- A copy of `data/human_queries_template.csv` saved as
  `data/human_queries.csv`, one row per card, filled in by the researcher.
- A timer.

## Procedure (per participant)

1. Read out the consent line (below). Continue only after a clear "yes";
   write `yes` in the `consent` column.
2. Ask the self-rating question; record it in `self_rated_knowledge`.
   Stop politely if the answer is 3 or higher.
3. Give the participant a pseudonymous ID (`P01`, `P02`, ...). **Never
   record names, e-mails or other personal details.**
4. For each card:
   1. Read the instructions (below) the first time.
   2. Show the note. The participant reads it **once**, for at most
      **60 seconds**.
   3. **Take the card away** (or move to a blank slide). The note must not
      be visible while they write.
   4. The participant writes or says their description; type it into the
      `query` column exactly as given (no corrections).
   5. Fill in `participant_id` and `date` (YYYY-MM-DD).
5. Do not suggest words, correct, or give examples beyond the instructions:
   any hint from the researcher becomes part of the query.

## Instructions to read out

> "You will see a short description of a wine written by a wine critic.
> Read it once; I will then take it away. Afterwards, tell me in one or two
> sentences how you imagine this wine tastes, in your own everyday words,
> as if you were describing it to a friend.
> Talk only about the taste and how it would feel in your mouth. You may say
> if it is red, white, rosé or sparkling. Please don't mention food,
> occasions or prices, and don't use names of grapes, places or producers,
> even if they were in the description. There are no wrong answers."

## Consent line

> "This is for a university project about wine recommendations. You will
> read a few wine descriptions and describe them in your own words; it takes
> about 10 minutes. Your answers are stored without your name and used only
> for this project. You can stop at any time without giving a reason. Do you
> agree to take part?"

## After the sessions

- Save `data/human_queries.csv` and commit it (it contains no personal data).
- `make compare-human` checks every hand-written query with the same
  vocabulary and leak checks as the LLM queries (without retries: a
  person's words are kept as written) and writes
  `results/llm_vs_human.csv` (HR@5 for the LLM queries, all human queries,
  and the human queries that pass the checks) and
  `results/human_queries_quality.csv`.
- If a participant used expert words, keep the query: how often that
  happens is itself a result.
