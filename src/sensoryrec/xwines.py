"""`make xwines`: the long tail and sparsity in real consumer ratings.

Analysis only (DECISIONS.md, 12 Nov 2026): no model is trained on this data
and nothing here feeds the recommenders. It answers one question for the
thesis: is the long tail seen in the expert reviews (Wine Enthusiast) also
there, or even stronger, when ordinary consumers rate wines?

Data: X-Wines (de Azambuja, Morais & Filipe 2023, Big Data Cogn. Comput.
7(1):20), an open dataset of 5-star consumer ratings, licence CC0.
Download ONE version by hand from the authors' Google Drive (link in
https://github.com/rogerioxavier/X-Wines, folder Dataset) and put its two CSV
files (wines and ratings) in data/raw/xwines/. Use the **Full** version: the
Test and Slim versions are small samples picked by the authors, so their
distribution is not the real one. The files are not committed; their names
and SHA-256 are recorded in results/xwines/data_manifest.json.

Outputs (results/xwines/)
- summary.csv            wines, users, ratings, density, rating values, dates
- long_tail.csv          one row per dataset and unit: Gini, share held by the
                         top 1/10/20%, share of units with few ratings.
                         X-Wines per wine, winery and user; Wine Enthusiast
                         per winery and variety, for comparison.
- ratings_by_type.csv    wines and ratings per wine type
- fig_lorenz.png         Lorenz curves, consumer ratings next to expert reviews
- fig_ratings_per_wine.png  ratings per wine, most rated first (log-log)
- data_manifest.json     which files were analysed
"""
import hashlib
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensoryrec.config import load_config, repo_path
from sensoryrec.eda import TOP_FRACTIONS, lorenz, top_share
from sensoryrec.metrics import gini

log = logging.getLogger(__name__)

WINE_COLUMNS = ["WineID", "Type", "Country", "WineryID"]
RATING_TYPES = {"UserID": "int32", "WineID": "int32", "Rating": "float32"}
# The Full version lists about 100,000 wines; Slim about 1,000 and Test 100.
FULL_MIN_WINES = 50_000


def find_files(folder):
    """The wines file and the ratings file in the folder (exactly one of each)."""
    wines = sorted(folder.glob("*wines*.csv"))
    ratings = sorted(folder.glob("*ratings*.csv"))
    if len(wines) != 1 or len(ratings) != 1:
        raise SystemExit(f"Put exactly one X-Wines wines CSV and one ratings CSV in {folder} "
                         f"(found {len(wines)} and {len(ratings)}). See the top of xwines.py.")
    return wines[0], ratings[0]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def date_range(ratings_path):
    """First and last rating date, read in chunks: the Full file has 21 million
    rows and the dates are text. Dates are written year-month-day, so the
    smallest text is the earliest date."""
    first, last = None, None
    for chunk in pd.read_csv(ratings_path, usecols=["Date"], chunksize=1_000_000):
        lo, hi = chunk["Date"].min(), chunk["Date"].max()
        first = lo if first is None else min(first, lo)
        last = hi if last is None else max(last, hi)
    return first, last


def counts_per(ratings, col, all_ids=None):
    """Number of ratings per value of `col`. With all_ids, units that were
    never rated are included with a count of 0."""
    counts = ratings[col].value_counts()
    if all_ids is not None:
        counts = counts.reindex(pd.Index(all_ids).unique(), fill_value=0)
    return counts


def long_tail_row(dataset, unit, event, counts, few):
    """How unevenly `event`s (ratings or reviews) are spread over units.
    few: a unit with at most this many events counts as "few" (the same
    number as the tail threshold of the thesis)."""
    counts = np.asarray(counts)
    row = {"dataset": dataset, "unit": unit, "event": event,
           "n_units": len(counts), "n_events": int(counts.sum()),
           "mean_per_unit": counts.mean(), "median_per_unit": float(np.median(counts)),
           "max_per_unit": int(counts.max()), "gini": gini(counts)}
    for f in TOP_FRACTIONS:
        row[f"top_{int(f * 100)}pct_share"] = top_share(counts, f)
    row["pct_units_with_1"] = (counts == 1).mean() * 100
    row[f"pct_units_le_{few}"] = (counts <= few).mean() * 100
    return row


def summary(wines, ratings):
    """Headline numbers of the ratings data, as (metric, value) rows."""
    pairs = len(ratings[["UserID", "WineID"]].drop_duplicates())
    n_users, n_rated = ratings["UserID"].nunique(), ratings["WineID"].nunique()
    rows = [
        ("wines_listed", len(wines)),
        ("wines_rated", n_rated),
        ("wineries", wines["WineryID"].nunique()),
        ("countries", wines["Country"].nunique()),
        ("users", n_users),
        ("ratings", len(ratings)),
        # a user can rate several vintages of the same wine: count each pair once
        ("user_wine_pairs", pairs),
        # share of all possible user-wine pairs that have a rating
        ("density_pct", pairs / (n_users * n_rated) * 100),
        ("mean_rating", float(ratings["Rating"].mean())),
        ("median_rating", float(ratings["Rating"].median())),
        ("pct_ratings_4_or_more", float((ratings["Rating"] >= 4).mean() * 100)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def ratings_by_type(wines, ratings):
    """Wines and ratings per wine type (red, white, ...), most rated first."""
    typed = ratings.merge(wines[["WineID", "Type"]], on="WineID", how="left")
    table = pd.DataFrame({"wines": wines["Type"].value_counts(),
                          "ratings": typed["Type"].value_counts(),
                          "mean_rating": typed.groupby("Type")["Rating"].mean()})
    table["ratings"] = table["ratings"].fillna(0).astype(int)
    table["pct_of_ratings"] = table["ratings"] / table["ratings"].sum() * 100
    return table.sort_values("ratings", ascending=False).rename_axis("type").reset_index()


def plot_lorenz(curves, path):
    """curves: label -> counts."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, counts in curves.items():
        x, y = lorenz(counts)
        ax.plot(x, y, label=f"{label} (Gini = {gini(counts):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect equality")
    ax.set_xlabel("Cumulative share of units (fewest first)")
    ax.set_ylabel("Cumulative share of ratings / reviews")
    ax.set_title("Lorenz curves: consumer ratings and expert reviews")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ratings_per_wine(counts, path):
    ranked = np.sort(np.asarray(counts))[::-1]
    ranked = ranked[ranked > 0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(np.arange(1, len(ranked) + 1), ranked)
    ax.set_xlabel("Wine rank (most rated first)")
    ax.set_ylabel("Number of ratings")
    ax.set_title("X-Wines: consumer ratings per wine")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    wines_path, ratings_path = find_files(repo_path(cfg["xwines"]["dir"]))
    catalogue_path = repo_path(cfg["paths"]["processed_dir"]) / "wines.parquet"
    if not catalogue_path.exists():
        raise SystemExit("Run `make data` first: the comparison needs the cleaned Wine Enthusiast catalogue.")
    few = cfg["tail"]["max_reviews"]
    out_dir = repo_path(cfg["xwines"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("reading %s and %s", wines_path.name, ratings_path.name)
    wines = pd.read_csv(wines_path, usecols=WINE_COLUMNS)
    ratings = pd.read_csv(ratings_path, usecols=list(RATING_TYPES), dtype=RATING_TYPES)
    unknown = int((~ratings["WineID"].isin(wines["WineID"])).sum())
    if unknown:
        raise SystemExit(f"{unknown} ratings are for wines missing from {wines_path.name}: "
                         "the two files are not from the same version.")
    full = len(wines) >= FULL_MIN_WINES
    if not full:
        log.warning("Only %d wines: this is the Test or Slim sample, picked by the authors. Its "
                    "distribution is not the real one; use the Full version for the thesis.", len(wines))
    expert = pd.read_parquet(catalogue_path, columns=["winery", "variety"])

    # Ratings per wine, per winery and per user; wines with no rating count as 0
    per_wine = counts_per(ratings, "WineID", wines["WineID"])
    winery_of = wines.set_index("WineID")["WineryID"]
    per_winery = per_wine.groupby(winery_of).sum()
    per_user = counts_per(ratings, "UserID")
    per_expert_winery = expert["winery"].value_counts()
    per_expert_variety = expert["variety"].value_counts()

    table = summary(wines, ratings)
    first, last = date_range(ratings_path)
    dates = pd.DataFrame([("first_rating", first), ("last_rating", last)], columns=["metric", "value"])
    pd.concat([table.round(4), dates]).to_csv(out_dir / "summary.csv", index=False)

    long_tail = pd.DataFrame([
        long_tail_row("X-Wines (consumers)", "wine", "ratings", per_wine, few),
        long_tail_row("X-Wines (consumers)", "winery", "ratings", per_winery, few),
        long_tail_row("X-Wines (consumers)", "user", "ratings", per_user, few),
        long_tail_row("Wine Enthusiast (experts)", "winery", "reviews", per_expert_winery, few),
        long_tail_row("Wine Enthusiast (experts)", "variety", "reviews", per_expert_variety, few),
    ])
    long_tail.round(4).to_csv(out_dir / "long_tail.csv", index=False)
    ratings_by_type(wines, ratings).round(3).to_csv(out_dir / "ratings_by_type.csv", index=False)

    plot_lorenz({"X-Wines: ratings per wine": per_wine,
                 "X-Wines: ratings per winery": per_winery,
                 "X-Wines: ratings per user": per_user,
                 "Wine Enthusiast: reviews per winery": per_expert_winery}, out_dir / "fig_lorenz.png")
    plot_ratings_per_wine(per_wine, out_dir / "fig_ratings_per_wine.png")

    manifest = {"source": "X-Wines, https://github.com/rogerioxavier/X-Wines (CC0)",
                "full_version": full,
                "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                          for p in (wines_path, ratings_path)]}
    (out_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("\n%s\n\n%s", table.round(4).to_string(index=False), long_tail.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
