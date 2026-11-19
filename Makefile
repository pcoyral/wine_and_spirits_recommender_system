PYTHON ?= python
export PYTHONPATH := src

.PHONY: data eda embed demo queries human-cards eval eval-dev refine compare-human test simulate eval-simulated xwines map-mockup app judge judge-agree xwines-cf embed-gemini gemini-subcatalogue faithfulness worked-example images photos photos-apply

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

# S3: 50 grounded explanations for annotation (needs GEMINI_API_KEY) -> results/faithfulness_sample.csv
faithfulness:
	$(PYTHON) -m sensoryrec.rag

# S3: the demo's complete worked example -> results/worked_example.{md,html,csv}
worked-example:
	$(PYTHON) -m sensoryrec.worked_example

# Gemini embeddings on the 10,000-wine sub-catalogue and every model compared
# there (needs GEMINI_API_KEY; resumable, ~12 daily runs) -> results/subcatalogue/
gemini-subcatalogue:
	$(PYTHON) -m sensoryrec.subcatalogue

# Gemini embeddings of every tasting note (needs GEMINI_API_KEY; resumable)
embed-gemini:
	$(PYTHON) -m sensoryrec.recommenders.semantic --only semantic_gemini

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

# LLM-as-a-judge on 150 simulated requests (needs GEMINI_API_KEY) -> results/judge/
judge:
	$(PYTHON) -m sensoryrec.judge

# Agreement between your hand scores and the judge -> results/judge/agreement.csv
judge-agree:
	$(PYTHON) -m sensoryrec.judge --agree

# Map MOCK-UP for Future Work -> docs/map_mockup.html (MODEL=tfidf to use another model)
map-mockup:
	$(PYTHON) -m sensoryrec.map_mockup $(if $(MODEL),--model $(MODEL),)

# Collaborative filtering vs content on X-Wines (cold start) -> results/xwines_cf/
xwines-cf:
	$(PYTHON) -m sensoryrec.xwines_cf

# Bottle images: X-Wines label photos (CC0) for matched wines -> data/images/, results/images_coverage.csv
images:
	$(PYTHON) -m sensoryrec.images

# Real bottle photos for the demo wines: candidates + review sheet (needs internet)
photos:
	$(PYTHON) -m sensoryrec.photos

# Use only the photos approved in data/images/photo_review.csv
photos-apply:
	$(PYTHON) -m sensoryrec.photos --apply

# Refinement: alpha on dev, evaluation on test -> results/refinement*.csv
refine:
	$(PYTHON) -m sensoryrec.refinement

# LLM vs hand-written queries on the shared test wines -> results/llm_vs_human.csv
compare-human:
	$(PYTHON) -m sensoryrec.compare_human

# Chat PROTOTYPE in the browser (S3, first version); needs requirements-app.txt
app:
	PYTHONPATH=src $(PYTHON) -m streamlit run app/chat.py

test:
	$(PYTHON) -m pytest -q
