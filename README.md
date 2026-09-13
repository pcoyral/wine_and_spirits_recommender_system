# SensoryRec

**A Natural Language-Driven Hybrid Recommender System for Wine and Spirits Discovery**

IE University BDBA Capstone Project — Paul, Class of 2027.

## Project Summary

SensoryRec recommends wine (and potentially spirits) using both structured product
attributes and natural-language descriptions of user taste preferences (e.g. "fruity
but not that fruity"). It targets younger, less wine-literate consumers as a
vocabulary-agnostic personalization layer, framed within an LVMH-style premiumization
business context.

## Architecture: the ablation ladder

Five recommender architectures, implemented as special cases of a single weighted
scoring function so the full ablation table can be produced from one parameterized
implementation:

| Model | Approach            | Notes                                    |
|-------|----------------------|-------------------------------------------|
| 0     | Popularity            | Non-personalized baseline                |
| 1     | Content-Based         | Structured attribute similarity          |
| 2     | TF-IDF                | Sparse text similarity on descriptions   |
| 3     | Sentence-BERT         | Dense semantic embeddings                |
| 4     | Hybrid                | Weighted combination of the above        |

Layers on top of the core models (per the technical methodology doc):
context-aware re-ranking and explainability.

## Repo structure

```
sensoryrec-capstone/
├── data/
│   ├── raw/            # Original Wine Enthusiast dataset (NOT committed — see below)
│   └── processed/       # Cleaned / feature-engineered data (NOT committed)
├── notebooks/           # EDA, prototyping, ablation experiments
├── src/
│   ├── models/           # popularity.py, content_based.py, tfidf.py, sbert.py, hybrid.py
│   ├── data_loader.py
│   └── metrics.py        # NDCG@5, serendipity, beyond-accuracy metrics
├── tests/
├── docs/                 # Reference documents: RQs, methodology, lit review roadmap
├── reports/              # EDA report, evaluation outputs
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd sensoryrec-capstone
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The full dataset (~130,000 Wine Enthusiast reviews) is sourced from Kaggle and is
**not committed to this repo** (see `.gitignore`). Download it separately and place
it at `data/raw/`. Document the exact source URL/version here once finalized, so the
project stays reproducible.

## Evaluation

No real user interaction data exists for this project. Evaluation uses a
self-retrieval protocol: paraphrased natural-language queries are matched against
held-out wines, with results validated by a sommelier domain expert. Key metrics:
NDCG@5, serendipity, and other beyond-accuracy measures.

## Status

Early scaffold — see `docs/` for the current reference documents (project
assessment, research questions & methodology, literature review roadmap, technical
methodology, development roadmap). Fill in `docs/` and start implementing
`src/models/popularity.py` (Model 0) first, per the development roadmap sequencing.
