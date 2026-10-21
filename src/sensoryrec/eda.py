"""S0 exploratory data analysis on the cleaned catalogue.

Run with `make eda` (needs `make data` first). Everything goes to results/eda/:
- summary.csv                  headline numbers (rows, wineries, Gini, ...)
- top_share.csv                share of reviews held by the top 1/10/20%
- taster_coverage.csv          reviews per taster
- vintage_years.csv            vintage years found in titles (scrape-date evidence)
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
    """Four-digit years 1900-2099 found in the wine title. The latest one is
    a lower bound for when the data was scraped."""
    years = df["title"].str.extract(r"\b(19\d{2}|20\d{2})\b")[0].dropna().astype(int)
    return years.value_counts().sort_index().rename_axis("year").reset_index(name="n_wines")


def summary_table(df, cleaning_log):
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
        "median_reviews_per_winery": float(winery.median()),
        "median_reviews_per_variety": float(variety.median()),
        "gini_winery": round(gini(winery), 4),
        "gini_variety": round(gini(variety), 4),
        "median_words": float(df["n_words"].median()),
        "pct_under_20_words": round((df["n_words"] < 20).mean() * 100, 2),
        "n_tasters": n_tasters,
        "pct_no_taster": round(df["taster_name_missing"].mean() * 100, 2),
        # One review per wine, so a taster x wine matrix has at most one
        # filled cell per column: density = reviews with taster / (tasters * wines)
        "taster_wine_matrix_density": len(tasters) / (n_tasters * len(df)),
        "reviews_per_wine": 1,
        "latest_vintage_in_titles": int(years["year"].max()),
    }
    # dtype=object keeps whole numbers as integers in the CSV
    return pd.DataFrame({"metric": list(rows.keys()), "value": pd.Series(list(rows.values()), dtype=object)})


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

    summary = summary_table(df, cleaning_log)
    tasters = taster_table(df)
    summary.to_csv(eda_dir / "summary.csv", index=False)
    top_share_table(df).to_csv(eda_dir / "top_share.csv", index=False)
    tasters.to_csv(eda_dir / "taster_coverage.csv", index=False)
    vintage_years(df).to_csv(eda_dir / "vintage_years.csv", index=False)

    plot_rank_frequency(df, eda_dir / "fig_rank_frequency.png")
    plot_lorenz(df, eda_dir / "fig_lorenz.png")
    plot_description_length(df, cfg["split"]["min_words"], eda_dir / "fig_description_length.png")
    plot_missingness(missing_df, eda_dir / "fig_missingness.png")
    plot_tasters(tasters, eda_dir / "fig_taster_coverage.png")
    log.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
