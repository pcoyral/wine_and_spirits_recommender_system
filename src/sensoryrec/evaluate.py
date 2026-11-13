"""`make eval`: score every model on the casual queries.

Input
- data/queries/{split}_queries.csv  columns wine_id, query: a casual,
  novice-style query written from the tasting note of wine_id (S2). The
  model should find that wine (self-retrieval).

Outputs (split "test"; for "dev" the names end in _dev)
- results/ablation.csv          one row per model, all metrics
- results/ablation_by_band.csv  the same per variety band (DECISIONS.md, S0)

`make eval-simulated` (split "simulated") uses the ~1,000 LLM-simulated
queries of data/queries/simulated_queries.csv instead. They have no correct
wine, so only the beyond-accuracy metrics are computed (coverage, tail share,
Gini, novelty, diversity), plus a coverage curve:
- results/simulated_coverage.csv        one row per model
- results/simulated_coverage_curve.csv  distinct wines recommended after n queries
- results/fig_simulated_coverage.png    the curve as a figure

Every model is fitted with the test wines held out, also when evaluating on
the dev split (CLAUDE.md rule 4). The metric definitions are in
DECISIONS.md; they were fixed before any results existed.
"""
import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sensoryrec import metrics
from sensoryrec.config import load_config, repo_path
from sensoryrec.recommenders import build_models, ild_vectors

log = logging.getLogger(__name__)


def winery_counts(wines, holdout_ids):
    """For every wine, the number of reviews of its winery, counted without
    the held-out wines (the same count as Popularity and the tail label)."""
    seen = wines[~wines["wine_id"].isin(holdout_ids)]
    return wines["winery"].map(seen["winery"].value_counts()).fillna(0).astype(int)


def item_popularity(wines, holdout_ids):
    """p for novelty: the wine's winery share of reviews, with one extra review
    added to every winery (add-one smoothing, DECISIONS.md):
        p = (winery reviews + 1) / (reviews counted + number of wineries)
    so p > 0 even for a winery whose only wines are held out. The shares of
    all wineries still add up to 1."""
    counts = winery_counts(wines, holdout_ids)
    n_seen = int((~wines["wine_id"].isin(holdout_ids)).sum())
    p = (counts + 1) / (n_seen + wines["winery"].nunique())
    return dict(zip(wines["wine_id"], p))


def beyond_accuracy(recs, ctx):
    """The metrics that need no correct wine: how widely and how evenly the
    recommendations spread over the catalogue, and how varied each list is."""
    row = {
        "coverage": metrics.catalogue_coverage(recs, ctx["n_wines"]),
        "winery_coverage": metrics.category_coverage(recs, ctx["winery_of"], len(ctx["all_wineries"])),
        "variety_coverage": metrics.category_coverage(recs, ctx["variety_of"], ctx["n_varieties"]),
        "tail_share": metrics.tail_share(recs, ctx["tail_ids"][ctx["tail_max"]]),
    }
    for t in ctx["robustness"]:
        row[f"tail_share_le{t}"] = metrics.tail_share(recs, ctx["tail_ids"][t])
    winery_recs = [[ctx["winery_of"][i] for i in rec] for rec in recs]
    row["gini_wine"] = metrics.recommendation_gini(recs, ctx["all_wines"])
    row["gini_winery"] = metrics.recommendation_gini(winery_recs, ctx["all_wineries"])
    row["novelty"] = metrics.novelty(recs, ctx["popularity"])
    row["ild"] = metrics.intra_list_diversity(recs, ctx["vectors"])
    return row


def metric_row(recs, targets, ctx):
    """All metrics for one set of recommendation lists with known targets."""
    k = ctx["k"]
    return {"n_queries": len(recs),
            f"hr@{k}": metrics.hit_rate_at_k(recs, targets, k),
            f"ndcg@{k}": metrics.ndcg_at_k(recs, targets, k),
            **beyond_accuracy(recs, ctx)}


def context(wines, holdout_ids, vectors, k, tail_max, robustness):
    """Everything the metrics need that is the same for every model."""
    counts = winery_counts(wines, holdout_ids)
    return {
        "k": k, "tail_max": tail_max, "robustness": robustness,
        "n_wines": len(wines),
        "tail_ids": {t: set(wines.loc[counts <= t, "wine_id"]) for t in [tail_max, *robustness]},
        "winery_of": dict(zip(wines["wine_id"], wines["winery"])),
        "variety_of": dict(zip(wines["wine_id"], wines["variety"])),
        "n_varieties": wines["variety"].nunique(),
        "all_wines": wines["wine_id"].tolist(),
        "all_wineries": wines["winery"].unique().tolist(),
        "popularity": item_popularity(wines, holdout_ids),
        "vectors": vectors,
    }


def evaluate(models, wines, queries, holdout_ids, vectors, k, tail_max, robustness):
    """Run every (fitted) model on every query and compute the metrics.

    queries: DataFrame with wine_id, query and band.
    vectors: wine_id -> SBERT embedding, used for ILD for every model.
    Returns (overall, by_band) DataFrames.
    """
    ctx = context(wines, holdout_ids, vectors, k, tail_max, robustness)
    queries = queries.reset_index(drop=True)
    targets = queries["wine_id"].tolist()
    overall, by_band = [], []
    for model in models:
        log.info("evaluating %s on %d queries", model.name, len(queries))
        recs = [model.recommend(q, k=k)["wine_id"].tolist() for q in queries["query"]]
        overall.append({"model": model.name, **metric_row(recs, targets, ctx)})
        for band, rows in queries.groupby("band").groups.items():
            band_recs = [recs[i] for i in rows]
            band_targets = [targets[i] for i in rows]
            by_band.append({"model": model.name, "band": band,
                            **metric_row(band_recs, band_targets, ctx)})
    return pd.DataFrame(overall), pd.DataFrame(by_band)


def coverage_curve(recs, n_wines, step):
    """Distinct wines recommended after the first n queries, for n = step,
    2*step, ... and the last query. A flat curve means the model keeps
    returning the same wines; a rising one means it keeps finding new ones."""
    seen, rows = set(), []
    for n, rec in enumerate(recs, start=1):
        seen.update(rec)
        if n % step == 0 or n == len(recs):
            rows.append({"n_queries": n, "distinct_wines": len(seen), "coverage": len(seen) / n_wines})
    return rows


def evaluate_simulated(models, wines, queries, holdout_ids, vectors, k, tail_max, robustness, step=50):
    """The simulated study: queries without a correct wine, so only the
    beyond-accuracy metrics. Returns (table, curve) DataFrames."""
    ctx = context(wines, holdout_ids, vectors, k, tail_max, robustness)
    table, curve = [], []
    for model in models:
        log.info("simulated study: %s on %d queries", model.name, len(queries))
        recs = [model.recommend(q, k=k)["wine_id"].tolist() for q in queries["query"]]
        # slots = recommendations made (queries x k): the most wines a model could
        # show. With 1,000 queries and k = 5 that is 5,000 of ~120,000 wines, so
        # coverage cannot exceed about 4%: compare models, not against 100%.
        slots = [wine_id for rec in recs for wine_id in rec]
        table.append({"model": model.name, "n_queries": len(recs), "slots": len(slots),
                      "distinct_wines": len(set(slots)), **beyond_accuracy(recs, ctx)})
        curve += [{"model": model.name, **row} for row in coverage_curve(recs, len(wines), step)]
    return pd.DataFrame(table), pd.DataFrame(curve)


def plot_coverage_curve(curve, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for model, g in curve.groupby("model", sort=False):
        ax.plot(g["n_queries"], g["distinct_wines"], marker="o", ms=3, label=model)
    ax.set_xlabel("Simulated queries so far")
    ax.set_ylabel("Distinct wines recommended")
    ax.set_title("How many different wines each model surfaces")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_simulated(cfg):
    path = repo_path(cfg["simulation"]["out"])
    if not path.exists():
        raise SystemExit(f"No simulated queries yet: run `make simulate` first ({path}). Nothing was evaluated.")
    queries = pd.read_csv(path)
    n_flagged = int(queries["flagged"].sum())
    if n_flagged:
        log.warning("leaving out %d flagged simulated queries", n_flagged)
    return queries[~queries["flagged"].astype(bool)].reset_index(drop=True)


def load_queries(cfg, split, split_ids):
    path = repo_path(cfg["eval"]["queries_dir"]) / f"{split}_queries.csv"
    if not path.exists():
        raise SystemExit(f"No casual queries yet: {path} is created in S2 "
                         "(columns wine_id, query). Nothing was evaluated.")
    queries = pd.read_csv(path)
    if "flagged" in queries:
        # Queries that still broke a rule after the retries are not used
        # (CLAUDE.md rule 5); how many is logged here and in queries_quality.csv.
        n_flagged = int(queries["flagged"].sum())
        if n_flagged:
            log.warning("leaving out %d flagged %s queries", n_flagged, split)
        queries = queries[~queries["flagged"].astype(bool)]
    unknown = set(queries["wine_id"]) - set(split_ids["wine_id"])
    if unknown:
        raise SystemExit(f"{len(unknown)} query wine_ids are not in the {split} set, e.g. {sorted(unknown)[:5]}")
    return queries.merge(split_ids[["wine_id", "band"]], on="wine_id")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["test", "dev", "simulated"], default="test")
    split = parser.parse_args().split

    cfg = load_config()
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")["wine_id"]
    if split == "simulated":
        queries = load_simulated(cfg)
    else:
        queries = load_queries(cfg, split, pd.read_csv(processed / f"{split}_ids.csv"))

    models = build_models(cfg)
    for model in models:
        model.fit(wines, holdout_ids=test_ids)
    settings = dict(holdout_ids=set(test_ids), vectors=ild_vectors(models), k=cfg["k"],
                    tail_max=cfg["tail"]["max_reviews"], robustness=cfg["eval"]["robustness_tail"])
    out_dir = repo_path(cfg["eval"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if split == "simulated":
        table, curve = evaluate_simulated(models, wines, queries, **settings)
        table.round(4).to_csv(out_dir / "simulated_coverage.csv", index=False)
        curve.round(6).to_csv(out_dir / "simulated_coverage_curve.csv", index=False)
        plot_coverage_curve(curve, out_dir / "fig_simulated_coverage.png")
        log.info("\n%s", table.round(4).to_string(index=False))
        return

    overall, by_band = evaluate(models, wines, queries, **settings)
    suffix = "" if split == "test" else "_dev"
    overall.round(4).to_csv(out_dir / f"ablation{suffix}.csv", index=False)
    by_band.round(4).to_csv(out_dir / f"ablation{suffix}_by_band.csv", index=False)
    log.info("\n%s", overall.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
