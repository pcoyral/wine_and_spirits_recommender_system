PYTHON ?= python
export PYTHONPATH := src

.PHONY: data eda test

# Definition of done for S0: `make data` gives the cleaned parquet, the
# test/dev IDs and all EDA figures. The two steps are recipe lines, not
# separate targets, so they always run in order (also with make -j).
data:
	$(PYTHON) -m sensoryrec.data
	$(PYTHON) -m sensoryrec.eda

# Rebuild only the EDA (needs `make data` once)
eda:
	$(PYTHON) -m sensoryrec.eda

test:
	$(PYTHON) -m pytest -q
