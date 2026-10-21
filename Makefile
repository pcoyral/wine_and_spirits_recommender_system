PYTHON ?= python
export PYTHONPATH := src

.PHONY: data eda embed demo queries human-cards eval eval-dev test

# Definition of done for S0: `make data` gives the cleaned parquet, the
# test/dev IDs and all EDA figures. The two steps are recipe lines, not
# separate targets, so they always run in order (also with make -j).
data:
	$(PYTHON) -m sensoryrec.data
	$(PYTHON) -m sensoryrec.eda

# Rebuild only the EDA (needs `make data` once)
eda:
	$(PYTHON) -m sensoryrec.eda

# Build the SBERT embedding cache (downloads the model on first use)
embed:
	$(PYTHON) -m sensoryrec.recommenders.semantic

# Top-5 of Popularity and Semantic for the queries in data/demo_queries.txt
demo:
	$(PYTHON) -m sensoryrec.demo

# Casual test + dev queries from the local LLM (Ollama must be running)
queries:
	$(PYTHON) -m sensoryrec.queries --split all

# Cards and answer template for the 30 hand-written queries
human-cards:
	$(PYTHON) -m sensoryrec.human

# All four models on the casual test queries -> results/ablation*.csv
eval:
	$(PYTHON) -m sensoryrec.evaluate --split test

# Same on the dev queries, for tuning -> results/ablation_dev*.csv
eval-dev:
	$(PYTHON) -m sensoryrec.evaluate --split dev

test:
	$(PYTHON) -m pytest -q
