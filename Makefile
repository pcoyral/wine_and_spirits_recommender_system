PYTHON ?= python
export PYTHONPATH := src

.PHONY: data clean-data eda test

# Definition of done for S0: `make data` gives the cleaned parquet, the
# test/dev IDs and all EDA figures.
data: clean-data eda

clean-data:
	$(PYTHON) -m sensoryrec.data

eda:
	$(PYTHON) -m sensoryrec.eda

test:
	$(PYTHON) -m pytest -q
