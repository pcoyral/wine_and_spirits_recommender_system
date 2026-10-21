# SensoryRec

**A content-based wine recommender for novices, delivered through a RAG chatbot.**

IE University BDBA Capstone Project — Paul, Class of 2027.

A novice describes what they like in everyday words ("something red that isn't
too heavy, for a barbecue"). SensoryRec retrieves matching wines from the Wine
Enthusiast Reviews dataset by comparing the query with professional tasting
notes using Sentence-BERT. An LLM then explains the picks, using only the
retrieved notes; it never ranks.

The research focus is **cold start under data sparsity** (no user–item
interactions, so collaborative filtering is not viable) and **long-tail
exposure** (does semantic matching surface more of the catalogue than
popularity?).

- `CLAUDE.md` — scope, rules and sprints (read this first)
- `DECISIONS.md` — methodological choices, with date and reason
- `docs/scope_v2.md` — scope after the 21 Oct 2026 supervisor meeting
- `legacy/` — code from before scope v2, kept for reuse

## Recommenders compared

| Model | Approach |
|---|---|
| Popularity | Non-personalised baseline |
| TF-IDF | Keyword match on tasting notes |
| Attribute | Structured fields (variety, country, price, points) |
| Semantic | SBERT `all-MiniLM-L6-v2` + cosine similarity |

## Setup

Requires **Python 3.11**.

```bash
# Create the venv OUTSIDE the project directory: if the repo lives in an
# iCloud-synced folder (~/Desktop, ~/Documents), iCloud offloads large
# packages like torch and imports block for minutes.
python3 -m venv ~/venvs/sensoryrec
source ~/venvs/sensoryrec/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Data

Download **Wine Reviews** (zackthoutt) from Kaggle,
https://www.kaggle.com/datasets/zynicide/wine-reviews, and put
`winemag-data-130k-v2.csv` in `data/raw/`. The file is not committed; its
SHA-256 is recorded in `results/eda/data_manifest.json` by `make data`.
`data/raw/03_Sample_Data.csv` (500 rows) is a schema reference only.

## Commands

| Command | What it does |
|---|---|
| `make data` | Clean the data, hold out test/dev wines, build all EDA tables and figures |
| `make eda` | Rebuild only the EDA (needs `make data` once) |
| `make test` | Run unit tests |

## Evaluation

Self-retrieval: held-out tasting notes are rewritten as casual, novice-style
queries, and each model must retrieve the original wine (HR@5, NDCG@5).
Long-tail and beyond-accuracy metrics: catalogue coverage, tail share, Gini,
novelty, intra-list diversity. All numbers come from `results/*.csv`.
