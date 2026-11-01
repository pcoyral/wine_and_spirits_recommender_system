# Mac checklist for today — get the live demo running

*Goal: `results/demo/demo_top5.md` on GitHub before the meeting.
Budget 1–2 hours, mostly waiting for downloads. Start early.*

## 0. Before you start (5 min)

- [ ] Write your **5 queries** in `data/demo_queries.txt`, one per line, the way
      a novice would say them (no wine vocabulary). Lines starting with `#` are
      ignored. Do this yourself: it keeps expert words out.
- [ ] `python3.11 --version` works. If not: `brew install python@3.11`.
- [ ] `make --version` works. If not: `xcode-select --install`.

## 1. Code and environment (15–30 min, mostly downloads)

```bash
cd ~/path/to/wine_and_spirits_recommender_system
git checkout main && git pull
python3.11 -m venv ~/venvs/sensoryrec
source ~/venvs/sensoryrec/bin/activate
pip install --upgrade pip
pip install -r requirements.txt     # torch is the big one
pip install -e .
make test                           # expect: 104 passed
```

## 2. The data (10 min)

1. Download **Wine Reviews** from Kaggle (https://www.kaggle.com/datasets/zynicide/wine-reviews;
   free account needed), unzip, and put `winemag-data-130k-v2.csv` in `data/raw/`.
2. Check it's the same file as the one behind the EDA:
   ```bash
   shasum -a 256 data/raw/winemag-data-130k-v2.csv
   grep raw_sha256 results/eda/data_manifest.json
   ```
   Same value → note "checksum matches Kaggle" in `DECISIONS.md` (Data source entry).
   Different → don't worry for tomorrow; tell me afterwards.
3. While on the Kaggle page: note the **scrape date / version date** and the
   **licence** (for `DECISIONS.md` and question 8).
4. `make data` (under a minute). Then `git status`:
   - `data/processed/test_ids.csv` or `dev_ids.csv` changed → **stop and tell
     me** (the test set must not change). Restore with `git checkout -- data/processed`.
   - Only `.png` files changed (different fonts/versions) → fine, don't commit them:
     `git checkout -- results/eda/*.png`.

## 3. The model — pin first, then embed (10–40 min)

The model version is part of the embedding cache's fingerprint, so **pin it
before `make embed`**, or the cache is built twice.

```bash
# download the model only (about 90 MB)
python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
cat ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main
```

- [ ] Copy that hash into `config.yaml` → `semantic: revision: <hash>`.
- [ ] `make embed` — encodes ~120,000 notes. Let it run (a progress bar shows).
      Out of memory? Set `semantic: batch_size: 16` and run it again.

## 4. The demo (2 min)

```bash
make demo
open results/demo/demo_top5.md      # or view it on GitHub after pushing
```

Read it once: does Popularity show the same wines for every query? Does
Semantic follow each query? Note one example you'll point at tomorrow.

## 5. Save it (5 min)

```bash
git add data/demo_queries.txt results/demo/ config.yaml DECISIONS.md
git commit -m "Demo queries, SBERT revision pinned, demo output for 2 Nov"
git push
```

Then check the files are visible on GitHub.

## If something fails

| Problem | What to do |
|---|---|
| `pip install` fails on torch | Check `python --version` says 3.11 inside the venv |
| `make: command not found` | `xcode-select --install` |
| Model download fails | Check the network; retry the one-line download command |
| `make embed` is very slow | Normal on an 8 GB Intel Mac; leave it running, plug in the charger |
| `make demo` says "No queries yet" | Your queries file has only `#` lines; add 5 plain lines |
| Anything else | Copy the error message to me |

**Fallback if the demo doesn't run in time:** present the EDA figures, the
code and the 104 passing tests, and say the live demo will follow by e-mail.
Don't run anything new during the meeting.

## Only if everything above is done (optional, 1–2 h)

First ablation table — risky on 8 GB, skip if tired:

1. Install Ollama, `ollama pull llama3.2:3b-instruct-q4_K_M`.
2. Pin `llm.digest` (ID column of `ollama list`) and `llm.ollama_version`
   (`ollama --version`) in `config.yaml`.
3. `make queries` → if `pct_flagged` in `results/queries_quality.csv` is high, stop.
4. `make eval` → `results/ablation.csv`. Commit `data/queries/`,
   `data/llm_cache/`, `results/`.

Do **not** show ablation numbers you haven't looked at carefully first.
