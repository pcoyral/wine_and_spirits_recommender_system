PYTHON ?= python
export PYTHONPATH := src

.PHONY: data eda embed demo test

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

test:
	$(PYTHON) -m pytest -q
