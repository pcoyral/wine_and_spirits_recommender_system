PYTHON ?= python
export PYTHONPATH := src

.PHONY: data eda embed demo queries human-cards eval eval-dev refine compare-human test simulate eval-simulated xwines

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

# Top-5 of all four models for the queries in data/demo_queries.txt
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

# ~1,000 simulated customer queries from Gemini (needs GEMINI_API_KEY); LIMIT=20 for a trial
simulate:
	$(PYTHON) -m sensoryrec.simulate $(if $(LIMIT),--limit $(LIMIT),)

# Coverage and diversity of every model on the simulated queries -> results/simulated_coverage*.csv
eval-simulated:
	$(PYTHON) -m sensoryrec.evaluate --split simulated

# Long tail and sparsity in consumer ratings (X-Wines, analysis only) -> results/xwines/
xwines:
	$(PYTHON) -m sensoryrec.xwines

# Refinement: alpha on dev, evaluation on test -> results/refinement*.csv
refine:
	$(PYTHON) -m sensoryrec.refinement

# LLM vs hand-written queries on the shared test wines -> results/llm_vs_human.csv
compare-human:
	$(PYTHON) -m sensoryrec.compare_human

test:
	$(PYTHON) -m pytest -q
