"""`make eval`: score all four models on the casual queries.

Input
- data/queries/{split}_queries.csv  columns wine_id, query: a casual,
  novice-style query written from the tasting note of wine_id (S2). The
  model should find that wine (self-retrieval).

Outputs (split "test"; for "dev" the names end in _dev)
- results/ablation.csv          one row per model, all metrics
- results/ablation_by_band.csv  the same per variety band (DECISIONS.md, S0)

Every model is fitted with the test wines held out, also when evaluating on
the dev split (CLAUDE.md rule 4). The metric definitions are in
DECISIONS.md; they were fixed before any results existed.
"""
import argparse
import logging

import pandas as pd

from sensoryrec import metrics
from sensoryrec.config import load_config, repo_path
from sensoryrec.recommenders import attribute, semantic, tfidf
from sensoryrec.recommenders.popularity import PopularityRecommender

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


def metric_row(recs, targets, ctx):
    """All metrics for one set of recommendation lists."""
    k = ctx["k"]
    row = {
        "n_queries": len(recs),
        f"hr@{k}": metrics.hit_rate_at_k(recs, targets, k),
        f"ndcg@{k}": metrics.ndcg_at_k(recs, targets, k),
        "coverage": metrics.catalogue_coverage(recs, ctx["n_wines"]),
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


def evaluate(models, wines, queries, holdout_ids, vectors, k, tail_max, robustness):
    """Run every (fitted) model on every query and compute the metrics.

    queries: DataFrame with wine_id, query and band.
    vectors: wine_id -> SBERT embedding, used for ILD for every model.
    Returns (overall, by_band) DataFrames.
    """
    counts = winery_counts(wines, holdout_ids)
    ctx = {
        "k": k, "tail_max": tail_max, "robustness": robustness,
        "n_wines": len(wines),
        "tail_ids": {t: set(wines.loc[counts <= t, "wine_id"]) for t in [tail_max, *robustness]},
        "winery_of": dict(zip(wines["wine_id"], wines["winery"])),
        "all_wines": wines["wine_id"].tolist(),
        "all_wineries": wines["winery"].unique().tolist(),
        "popularity": item_popularity(wines, holdout_ids),
        "vectors": vectors,
    }
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
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    split = parser.parse_args().split

    cfg = load_config()
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")["wine_id"]
    split_ids = pd.read_csv(processed / f"{split}_ids.csv")
    queries = load_queries(cfg, split, split_ids)

    models = [
        PopularityRecommender(),
        tfidf.from_config(cfg),
        attribute.from_config(cfg),
        semantic.from_config(cfg),
    ]
    for model in models:
        model.fit(wines, holdout_ids=test_ids)
    vectors = models[-1].vectors()

    overall, by_band = evaluate(models, wines, queries, set(test_ids), vectors, k=cfg["k"],
                                tail_max=cfg["tail"]["max_reviews"],
                                robustness=cfg["eval"]["robustness_tail"])
    out_dir = repo_path(cfg["eval"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if split == "test" else "_dev"
    overall.round(4).to_csv(out_dir / f"ablation{suffix}.csv", index=False)
    by_band.round(4).to_csv(out_dir / f"ablation{suffix}_by_band.csv", index=False)
    log.info("\n%s", overall.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
