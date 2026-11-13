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
| Semantic (extra) | The same recommender with another embedding model (`extra_embedding_models` in `config.yaml`; first `BAAI/bge-small-en-v1.5`) |

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
| `make embed` | Build the embedding cache of every embedding model (downloads the models on first use) |
| `make demo` | Top-5 of every model for the queries in `data/demo_queries.txt` |
| `make queries` | LLM casual queries for the test and dev wines (Ollama running) → `data/queries/`, `results/queries_quality.csv` |
| `make human-cards` | Note cards and answer template for the 30 hand-written queries |
| `make refine` | Refinement: α on dev, evaluation on test → `results/refinement*.csv` |
| `make compare-human` | LLM vs hand-written queries → `results/llm_vs_human.csv` |
| `make eval` | Every model on the casual test queries → `results/ablation.csv` (needs S2 queries) |
| `make eval-dev` | The same on the dev queries, for tuning → `results/ablation_dev.csv` |
| `make simulate` | ~1,000 simulated customer queries written by Gemini (needs `GEMINI_API_KEY`; `LIMIT=20` for a trial) → `data/queries/simulated_queries.csv` |
| `make eval-simulated` | Coverage and diversity of every model on the simulated queries → `results/simulated_coverage*.csv` |
| `make xwines` | Long tail and sparsity in consumer ratings (X-Wines, analysis only; put the Full version's two CSVs in `data/raw/xwines/`) → `results/xwines/` |
| `make map-mockup` | Map MOCK-UP for Future Work (real origins of the top-5, invented shops) → `docs/map_mockup.html`; see `docs/map_libraries.md` |
| `make test` | Run unit tests |

## Evaluation

Self-retrieval: held-out tasting notes are rewritten as casual, novice-style
queries, and each model must retrieve the original wine (HR@5, NDCG@5).
Long-tail and beyond-accuracy metrics: catalogue coverage, tail share, Gini,
novelty, intra-list diversity. All numbers come from `results/*.csv`.

Simulated-query study: about 1,000 customer requests, built from seeded
combinations of wishes (`data/simulation_facets.yaml`) and phrased by an LLM.
They have no correct wine, so they measure only how widely and evenly each
model recommends (coverage of wines, wineries and varieties, tail share,
Gini, novelty, diversity), not accuracy.

## Running in the cloud (GitHub Actions)

Two manual workflows (Actions tab → pick one → "Run workflow") do the heavy
steps on GitHub's machine and push their output to a branch:

| Workflow | What it runs | Output branch |
|---|---|---|
| Demo | `make data`, `make embed`, `make demo` | `demo-results` |
| Simulated queries | `make simulate`, `make embed`, `make eval-simulated` | `simulation-results` |

"Simulated queries" needs the repository secret `GEMINI_API_KEY` (Settings →
Secrets and variables → Actions) and `simulation.llm.model` set in
`config.yaml`. The key is never written to the repo. If the free quota runs
out, the answers so far are saved; run the workflow again to continue.
