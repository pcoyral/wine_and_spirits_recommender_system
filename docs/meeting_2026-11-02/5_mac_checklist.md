# Mac checklist for today: get the live demo running

*Goal: `results/demo/demo_top5.md` on GitHub before the meeting.
Allow 1 to 2 hours, mostly waiting for downloads, so start early.*

## 0. Before you start (5 min)

- [ ] Write your 5 queries in `data/demo_queries.txt`, one per line, the way a
      novice would say them (no wine vocabulary). Lines starting with `#` are
      ignored. Write them yourself so that no expert words slip in.
- [ ] Check that `python3.11 --version` works. If it doesn't, run `brew install python@3.11`.
- [ ] Check that `make --version` works. If it doesn't, run `xcode-select --install`.

## 1. Code and environment (15 to 30 min, mostly downloads)

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

1. Download Wine Reviews from Kaggle (https://www.kaggle.com/datasets/zynicide/wine-reviews;
   you need a free account), unzip it, and put `winemag-data-130k-v2.csv` in `data/raw/`.
2. Check that it's the same file the EDA used:
   ```bash
   shasum -a 256 data/raw/winemag-data-130k-v2.csv
   grep raw_sha256 results/eda/data_manifest.json
   ```
   If the two values match, write "checksum matches Kaggle" in the Data source
   entry of `DECISIONS.md`. If they differ, leave it for tomorrow and tell me
   afterwards.
3. While you're on the Kaggle page, note the scrape or version date and the
   licence (for `DECISIONS.md` and question 8).
4. Run `make data` (under a minute), then `git status`:
   - If `data/processed/test_ids.csv` or `dev_ids.csv` changed, stop and tell
     me, because the test set must not change. Restore them with
     `git checkout -- data/processed`.
   - If only `.png` files changed (different fonts or library versions), that's
     fine. Don't commit them: `git checkout -- results/eda/*.png`.

## 3. The model: pin it, then embed (10 to 40 min)

The model version is part of the embedding cache's fingerprint. Pin it before
running `make embed`, or the cache gets built twice.

```bash
# download the model only (about 90 MB)
python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
cat ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main
```

- [ ] Copy that hash into `config.yaml` as `semantic: revision: <hash>`.
- [ ] Run `make embed`, which encodes about 120,000 notes. Let it run; a
      progress bar shows how far it has got. If it runs out of memory, set
      `semantic: batch_size: 16` and run it again.

## 4. The demo (2 min)

```bash
make demo
open results/demo/demo_top5.md      # or view it on GitHub after pushing
```

Read it once. Does Popularity show the same wines for every query? Does
Semantic follow each query? Pick one example to point at tomorrow.

## 5. Save it (5 min)

```bash
git add data/demo_queries.txt results/demo/ config.yaml DECISIONS.md
git commit -m "Demo queries, SBERT revision pinned, demo output for 2 Nov"
git push
```

Then check that the files show up on GitHub.

## If something fails

| Problem | What to do |
|---|---|
| `pip install` fails on torch | Check that `python --version` says 3.11 inside the venv |
| `make: command not found` | `xcode-select --install` |
| Model download fails | Check the network and retry the one-line download command |
| `make embed` is very slow | That's normal on an 8 GB Intel Mac. Leave it running with the charger plugged in |
| `make demo` says "No queries yet" | Your queries file only has `#` lines. Add 5 plain lines |
| Anything else | Copy the error message to me |

Fallback if the demo doesn't run in time: present the EDA figures, the code
and the 104 passing tests, and say you'll send the live demo by e-mail. Don't
run anything new during the meeting.

## Optional, only if everything above is done (1 to 2 h)

The first ablation table. It's risky on 8 GB, so skip it if you're tired.

1. Install Ollama and run `ollama pull llama3.2:3b-instruct-q4_K_M`.
2. Pin `llm.digest` (the ID column of `ollama list`) and `llm.ollama_version`
   (`ollama --version`) in `config.yaml`.
3. Run `make queries`. If `pct_flagged` in `results/queries_quality.csv` is
   high, stop there.
4. Run `make eval` to get `results/ablation.csv`. Commit `data/queries/`,
   `data/llm_cache/` and `results/`.

Don't show any ablation numbers you haven't looked at carefully first.
