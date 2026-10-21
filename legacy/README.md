# legacy/

Code from before scope v2 (21 Oct 2026), kept for reference and reuse.
It is not part of the pipeline and its tests are not run by `make test`.

| File | Status under scope v2 |
|---|---|
| `src/models/base.py` | `fit` / `score` / `recommend` pattern to reuse in S1 (`recommenders/base.py`) |
| `src/metrics.py` | `ndcg_at_k` is correct; reuse in S1 `metrics.py` |
| `src/models/popularity.py`, `tfidf_model.py`, `sbert_model.py`, `content_based.py` | Starting points for the S1 recommenders |
| `src/models/hybrid.py` | OUT of scope (no CF to hybridise) |
| `src/data_loader.py` | Loading superseded by `src/sensoryrec/data.py`; LVMH labelling kept here |
| `notebooks/01_eda.ipynb` | Superseded by `src/sensoryrec/eda.py` |

To run the old tests: `cd legacy && python -m pytest -q tests` (with `legacy/` on the path).
