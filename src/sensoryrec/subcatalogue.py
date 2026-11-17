"""Gemini embeddings on a 10,000-wine sub-catalogue (student's decision C,
17 Nov 2026; DECISIONS.md).

The free Gemini quota embeds about 1,000 texts a day, so all ~120,000 notes
would take about four months. Instead every model is compared on the same
smaller catalogue:
- every test and dev wine (the targets of the S2 queries), plus
- other wines drawn at random with the seed, up to 10,000.
The list is written once to data/processed/subcatalogue_ids.csv (committed),
so every run uses the same wines.

`make gemini-subcatalogue` (needs GEMINI_API_KEY; resumable, ~12 daily runs)
1. embeds the notes of the sub-catalogue with Gemini, 100 per request; the
   finished batches are kept, so a run stopped by the daily quota continues;
2. fits every model of the main comparison (Popularity, TF-IDF, Attribute,
   Sentence-BERT, bge-small) plus Gemini on the sub-catalogue, test wines
   held out as everywhere else;
3. evaluates them on the S2 test queries and the ~1,000 simulated queries
   (Gemini embeds each query once; cached in data/llm_cache/) and writes
   results/subcatalogue/ablation*.csv and simulated_coverage*.csv.

Tail share and novelty count winery reviews in the FULL catalogue, so a
"tail winery" means the same as in the main results. Coverage is relative to
the 10,000 wines. Numbers are comparable between models of this study, not
with the full-catalogue tables.
"""
import copy
import logging

import pandas as pd

from sensoryrec.config import load_config, repo_path
from sensoryrec.evaluate import (evaluate, evaluate_simulated, k_values, load_queries,
                                 load_simulated, plot_coverage_curve)
from sensoryrec.recommenders import build_models, ild_vectors

log = logging.getLogger(__name__)


def draw_ids(wines, keep_ids, size, seed):
    """All `keep_ids` plus wines drawn at random (seeded) up to `size`,
    in catalogue order."""
    keep = wines["wine_id"].isin(set(keep_ids))
    n_more = size - int(keep.sum())
    if n_more < 0:
        raise ValueError(f"{int(keep.sum())} wines must be kept, more than size {size}")
    more = wines.loc[~keep, "wine_id"].sample(n=n_more, random_state=seed)
    chosen = set(wines.loc[keep, "wine_id"]) | set(more)
    return wines.loc[wines["wine_id"].isin(chosen), "wine_id"].tolist()


def subcatalogue_ids(cfg, wines):
    """The sub-catalogue: read from its file, or drawn and written the first time."""
    c = cfg["subcatalogue"]
    path = repo_path(c["ids"])
    if path.exists():
        ids = pd.read_csv(path)["wine_id"].tolist()
        missing = set(ids) - set(wines["wine_id"])
        if missing:
            raise SystemExit(f"{len(missing)} sub-catalogue wines are not in wines.parquet, e.g. {sorted(missing)[:5]}")
        return ids
    processed = repo_path(cfg["paths"]["processed_dir"])
    keep = pd.concat([pd.read_csv(processed / f"{s}_ids.csv")["wine_id"] for s in ("test", "dev")])
    ids = draw_ids(wines, keep, c["size"], cfg["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"wine_id": ids}).to_csv(path, index=False)
    log.info("drew the sub-catalogue: %d wines -> %s", len(ids), path)
    return ids


def subcatalogue_config(cfg):
    """The main comparison's models, with Gemini switched on and their
    embedding caches in a separate folder (the full-catalogue caches stay
    as they are)."""
    cfg = copy.deepcopy(cfg)
    cfg["semantic"]["cache_dir"] = cfg["subcatalogue"]["embeddings_dir"]
    for m in cfg.get("extra_embedding_models") or []:
        m["enabled"] = True
    return cfg


def fit_models(models, wines, test_ids, queries=()):
    """Gemini first, and its query embeddings right after: if the daily quota
    stops it, nothing else is computed that day."""
    order = sorted(models, key=lambda m: m.name != "semantic_gemini")
    for model in order:
        model.fit(wines, holdout_ids=test_ids)
        if model.name == "semantic_gemini" and len(queries):
            model.encode_queries([model.query_prefix + q for q in queries])
    return models


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    processed = repo_path(cfg["paths"]["processed_dir"])
    full = pd.read_parquet(processed / "wines.parquet")
    ids = set(subcatalogue_ids(cfg, full))
    wines = full[full["wine_id"].isin(ids)].reset_index(drop=True)
    test_ids = pd.read_csv(processed / "test_ids.csv")["wine_id"]

    test_queries = load_queries(cfg, "test", pd.read_csv(processed / "test_ids.csv"))
    simulated = load_simulated(cfg)
    all_queries = [*test_queries["query"], *simulated["query"]]
    models = fit_models(build_models(subcatalogue_config(cfg)), wines, test_ids, all_queries)
    log.info("all %d models fitted on %d wines", len(models), len(wines))

    out = repo_path(cfg["subcatalogue"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for k in k_values(cfg):
        tag = "" if k == cfg["k"] else f"_k{k}"
        settings = dict(holdout_ids=set(test_ids), vectors=ild_vectors(models), k=k,
                        tail_max=cfg["tail"]["max_reviews"], robustness=cfg["eval"]["robustness_tail"],
                        reference=full)
        overall, by_band = evaluate(models, wines, test_queries, **settings)
        overall.round(4).to_csv(out / f"ablation{tag}.csv", index=False)
        by_band.round(4).to_csv(out / f"ablation{tag}_by_band.csv", index=False)
        table, curve = evaluate_simulated(models, wines, simulated, **settings)
        table.round(4).to_csv(out / f"simulated_coverage{tag}.csv", index=False)
        curve.round(6).to_csv(out / f"simulated_coverage_curve{tag}.csv", index=False)
        plot_coverage_curve(curve, out / f"fig_simulated_coverage{tag}.png", k)
        log.info("k = %d\n%s\n%s", k, overall.round(4).to_string(index=False), table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
