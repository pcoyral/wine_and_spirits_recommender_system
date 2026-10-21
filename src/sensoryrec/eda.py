"""S0 exploratory data analysis on the cleaned catalogue.

Run with `make eda` (needs `make data` first). Everything goes to results/eda/:
- summary.csv                  headline numbers (rows, wineries, Gini, ...)
- top_share.csv                share of reviews held by the top 1/10/20%
- taster_coverage.csv          reviews per taster
- vintage_years.csv            vintage years found in titles (scrape-date evidence)
- tail_summary.csv             size of the long tail under config.yaml's definition
- tail_options.csv             the tail definitions considered (DECISIONS.md)
- fig_rank_frequency.png       long-tail plot, wineries and varieties
- fig_lorenz.png               Lorenz curves with Gini
- fig_description_length.png   words per tasting note
- fig_missingness.png          % missing per column (from data.py)
- fig_taster_coverage.png      reviews per taster
"""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensoryrec.config import load_config, repo_path
from sensoryrec.data import metric_table, tail_mask
from sensoryrec.metrics import gini

log = logging.getLogger(__name__)

TOP_FRACTIONS = [0.01, 0.10, 0.20]


def reviews_per(df, col):
    """Number of reviews per value of `col`, largest first."""
    return df[col].value_counts()


def lorenz(counts):
    """Points of the Lorenz curve: cumulative share of items (x) against
    cumulative share of reviews (y), items sorted from fewest to most."""
    x = np.sort(np.asarray(counts, dtype=float))
    cum = np.concatenate([[0], np.cumsum(x)]) / x.sum()
    items = np.linspace(0, 1, len(x) + 1)
    return items, cum


def top_share(counts, fraction):
    """Share of all reviews held by the top `fraction` of items.
    The number of top items is rounded up, so it is never zero."""
    counts = np.sort(np.asarray(counts))[::-1]
    k = int(np.ceil(len(counts) * fraction))
    return counts[:k].sum() / counts.sum()


def top_share_table(df):
    rows = []
    for col in ["winery", "variety"]:
        counts = reviews_per(df, col)
        for f in TOP_FRACTIONS:
            rows.append({"entity": col, "top_fraction": f,
                         "n_items": int(np.ceil(len(counts) * f)),
                         "share_of_reviews": round(top_share(counts, f), 4)})
    return pd.DataFrame(rows)


def taster_table(df):
    """Reviews per taster. 'Unknown' means the review has no taster."""
    t = df["taster_name"].value_counts().rename_axis("taster").reset_index(name="n_reviews")
    t["share"] = (t["n_reviews"] / len(df)).round(4)
    return t


def vintage_years(df):
    """Vintage year of each wine, read from its title. Titles look like
    "Winery 2013 Name Variety (Region)", so we cut off the winery name and
    take the first four-digit year (1900-2099) after it. Taking simply the
    first or last year in the title goes wrong for names such as
    "1912 Winemakers 2011 ..." or "Kirsten 2013 1904 Spätlese ...".
    The latest vintage is a lower bound for when the data was scraped."""
    rest = [t[len(w):] if t.startswith(w) else t for t, w in zip(df["title"], df["winery"])]
    found = pd.Series(rest, index=df.index).str.extract(r"\b(19\d{2}|20\d{2})\b")[0]
    years = found.dropna().astype(int)
    return years.value_counts().sort_index().rename_axis("year").reset_index(name="n_wines")


def summary_table(df, cleaning_log, min_words):
    winery = reviews_per(df, "winery")
    variety = reviews_per(df, "variety")
    tasters = df.loc[~df["taster_name_missing"], "taster_name"]
    years = vintage_years(df)
    n_tasters = tasters.nunique()
    rows = {
        "raw_rows": int(cleaning_log["rows_before"].iloc[0]),
        "clean_rows": len(df),
        "n_wineries": len(winery),
        "n_varieties": len(variety),
        "n_countries": df.loc[df["country"] != "Unknown", "country"].nunique(),
        "wineries_with_1_review": int((winery == 1).sum()),
        "varieties_with_1_review": int((variety == 1).sum()),
        "max_reviews_per_winery": int(winery.max()),
        "max_reviews_per_variety": int(variety.max()),
        "median_reviews_per_winery": float(winery.median()),
        "median_reviews_per_variety": float(variety.median()),
        "gini_winery": round(gini(winery), 4),
        "gini_variety": round(gini(variety), 4),
        "median_words": float(df["n_words"].median()),
        f"pct_under_{min_words}_words": round((df["n_words"] < min_words).mean() * 100, 2),
        "n_tasters": n_tasters,
        "pct_no_taster": round(df["taster_name_missing"].mean() * 100, 2),
        # One review per wine, so a taster x wine matrix has at most one
        # filled cell per column: density = reviews with taster / (tasters * wines)
        "taster_wine_matrix_density": len(tasters) / (n_tasters * len(df)),
        "reviews_per_wine": 1,
        "latest_vintage_in_titles": int(years["year"].max()),
    }
    return metric_table(rows)


def tail_summary(df, test_ids, tail_cfg):
    """How big the long tail is under the definition in config.yaml."""
    unit, k = tail_cfg["unit"], tail_cfg["max_reviews"]
    tail = tail_mask(df, unit, k)
    counts = reviews_per(df, unit)
    rows = {
        "definition": f"{unit} with <= {k} reviews",
        f"tail_{unit}_count": int((counts <= k).sum()),
        f"tail_{unit}_pct": round((counts <= k).mean() * 100, 2),
        "tail_wines": int(tail.sum()),
        "tail_wines_pct": round(tail.mean() * 100, 2),
        "test_wines_in_tail": int(tail[df["wine_id"].isin(test_ids)].sum()),
    }
    return metric_table(rows)


def tail_options(df, test_ids):
    """The long-tail definitions compared before choosing one (DECISIONS.md).
    Each becomes 'tail = at most max_reviews reviews', so all go through
    tail_mask:
    - A: head = the top 20% of items by review count
    - C: head = the most-reviewed items that together hold 20% / 50% of reviews
    - B: fixed thresholds
    """
    rows = []
    for unit, fixed in [("winery", [5, 10, 20]), ("variety", [10, 100, 1000])]:
        counts = reviews_per(df, unit)  # most reviewed first
        rules = []
        head_min = counts.iloc[int(np.ceil(0.2 * len(counts))) - 1]
        rules.append(("A", "head = top 20% of items", head_min - 1))
        share = counts.cumsum() / counts.sum()
        for p in [0.2, 0.5]:
            head_min = counts[share >= p].iloc[0]
            rules.append(("C", f"head = items holding first {p:.0%} of reviews", head_min - 1))
        rules += [("B", f"at most {k} reviews", k) for k in fixed]

        for option, rule, max_reviews in rules:
            tail = tail_mask(df, unit, max_reviews)
            n_tail_items = int((counts <= max_reviews).sum())
            rows.append({
                "unit": unit, "option": option, "rule": rule,
                "tail_max_reviews": int(max_reviews),
                "tail_items": n_tail_items,
                "tail_items_pct": round(n_tail_items / len(counts) * 100, 1),
                "tail_wines_pct": round(tail.mean() * 100, 1),
                "test_wines_in_tail": int(tail[df["wine_id"].isin(test_ids)].sum()),
            })
    return pd.DataFrame(rows)


def plot_rank_frequency(df, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in ["winery", "variety"]:
        counts = reviews_per(df, col).to_numpy()
        ax.loglog(np.arange(1, len(counts) + 1), counts, label=col)
    ax.set_xlabel("Rank (1 = most reviewed)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Reviews per winery and per variety (log-log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_lorenz(df, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for col in ["winery", "variety"]:
        counts = reviews_per(df, col)
        x, y = lorenz(counts)
        ax.plot(x, y, label=f"{col} (Gini = {gini(counts):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect equality")
    ax.set_xlabel("Cumulative share of items (fewest reviews first)")
    ax.set_ylabel("Cumulative share of reviews")
    ax.set_title("Lorenz curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_description_length(df, min_words, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["n_words"], bins=80)
    ax.axvline(min_words, color="k", ls="--", lw=1, label=f"{min_words} words (test/dev minimum)")
    ax.set_xlabel("Words in tasting note")
    ax.set_ylabel("Number of wines")
    ax.set_title("Description length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_missingness(missing_df, path):
    m = missing_df.sort_values("pct_missing")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(m["column"], m["pct_missing"])
    ax.set_xlabel("% missing")
    ax.set_title("Missing values per column (after row cleaning)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tasters(tasters, path):
    t = tasters.sort_values("n_reviews")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(t["taster"], t["n_reviews"])
    ax.set_xlabel("Number of reviews")
    ax.set_title("Reviews per taster ('Unknown' = no taster)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    eda_dir = repo_path(cfg["paths"]["eda_dir"])
    df = pd.read_parquet(repo_path(cfg["paths"]["processed_dir"]) / "wines.parquet")
    cleaning_log = pd.read_csv(eda_dir / "cleaning_log.csv")
    missing_df = pd.read_csv(eda_dir / "missingness.csv")

    summary = summary_table(df, cleaning_log, cfg["split"]["min_words"])
    tasters = taster_table(df)
    summary.to_csv(eda_dir / "summary.csv", index=False)
    top_share_table(df).to_csv(eda_dir / "top_share.csv", index=False)
    tasters.to_csv(eda_dir / "taster_coverage.csv", index=False)
    vintage_years(df).to_csv(eda_dir / "vintage_years.csv", index=False)
    test_ids = pd.read_csv(repo_path(cfg["paths"]["processed_dir"]) / "test_ids.csv")["wine_id"]
    tail = tail_summary(df, test_ids, cfg["tail"])
    tail.to_csv(eda_dir / "tail_summary.csv", index=False)
    tail_options(df, test_ids).to_csv(eda_dir / "tail_options.csv", index=False)

    plot_rank_frequency(df, eda_dir / "fig_rank_frequency.png")
    plot_lorenz(df, eda_dir / "fig_lorenz.png")
    plot_description_length(df, cfg["split"]["min_words"], eda_dir / "fig_description_length.png")
    plot_missingness(missing_df, eda_dir / "fig_missingness.png")
    plot_tasters(tasters, eda_dir / "fig_taster_coverage.png")
    log.info("\n%s", summary.to_string(index=False))
    log.info("\n%s", tail.to_string(index=False))


if __name__ == "__main__":
    main()
