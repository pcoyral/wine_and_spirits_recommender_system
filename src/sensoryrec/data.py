"""S0 data pipeline: load the raw CSV, clean it, hold out test and dev wines.

Run with `make data` (or `python -m sensoryrec.data`).

Outputs
- data/processed/wines.parquet   cleaned catalogue (test/dev wines included,
                                 because they are the targets to retrieve)
- data/processed/test_ids.csv    200 held-out wines
- data/processed/dev_ids.csv     100 wines for tuning settings
- results/eda/cleaning_log.csv   rows removed by each cleaning step
- results/eda/missingness.csv    % missing per column, before filling
- results/eda/duplicate_check.csv  are removed duplicates the same wine?
- results/eda/duplicate_different_wines.csv  duplicates attached to a different wine
- results/eda/data_manifest.json raw file checksum and split sizes
"""
import hashlib
import json
import logging

import numpy as np
import pandas as pd

from sensoryrec.config import load_config, repo_path

log = logging.getLogger(__name__)

# Text columns where a missing value becomes "Unknown" (with a flag for the
# ones we use later). Price is NOT filled: an invented price would distort
# the attribute baseline and "under 20" filters.
FILL_UNKNOWN = ["country", "province", "region_1", "region_2", "designation", "taster_name"]
FLAG_MISSING = ["price", "region_1", "taster_name"]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path):
    """Read the Kaggle CSV. The first, unnamed column is the original row
    index; we keep it as wine_id so IDs stay the same on every run."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "wine_id"})
    else:
        df.insert(0, "wine_id", np.arange(len(df)))
    return df


def word_count(text):
    return text.str.split().str.len()


def missingness(df):
    """Percentage of missing values per column."""
    pct = df.isna().mean().mul(100).round(2)
    return pct.rename("pct_missing").rename_axis("column").reset_index()


def clean(df):
    """Clean the raw table. Returns (clean_df, log_df, missing_df, dup_pairs).

    Each step records how many rows it removed, so every count in the
    thesis can be traced back to this function. dup_pairs holds each removed
    duplicate next to the copy that was kept (see duplicate_check).
    """
    steps = []

    def record(step, before, after):
        steps.append({"step": step, "rows_before": before,
                      "rows_removed": before - after, "rows_after": after})
        log.info("%-28s removed %6d rows, %6d left", step, before - after, after)

    record("load raw", len(df), len(df))

    df = df.copy()
    df["description"] = df["description"].astype("string").str.strip()

    n = len(df)
    df = df[df["description"].notna() & (df["description"] != "")]
    record("drop empty description", n, len(df))

    # Missing variety goes before de-duplication, so a usable duplicate is
    # never thrown away in favour of a copy without a variety.
    n = len(df)
    df = df[df["variety"].notna()]
    record("drop missing variety", n, len(df))

    n = len(df)
    is_dup = df.duplicated(subset="description", keep="first")
    dup_pairs = duplicate_pairs(df, is_dup)
    df = df[~is_dup]
    record("drop duplicate description", n, len(df))

    # Missingness is measured here, after removing rows but before filling,
    # so it describes the data we actually use.
    missing_df = missingness(df)

    df = df.drop(columns=["taster_twitter_handle"], errors="ignore")
    for col in FLAG_MISSING:
        df[f"{col}_missing"] = df[col].isna()
    for col in FILL_UNKNOWN:
        df[col] = df[col].fillna("Unknown")

    df["n_words"] = word_count(df["description"])
    df = df.reset_index(drop=True)
    return df, pd.DataFrame(steps), missing_df, dup_pairs


DUP_COLS = ["title", "winery", "points", "price"]


def duplicate_pairs(df, is_dup):
    """Each removed duplicate (is_dup True) next to the copy that is kept."""
    kept = df[~is_dup].set_index("description")[DUP_COLS]
    return df[is_dup].join(kept, on="description", rsuffix="_kept")


def duplicate_check(pairs):
    """Are the duplicate notes the same wine scraped twice?

    Same title, winery, points and price as the kept copy = the same wine.
    Returns (summary_df, different_df): counts, and the pairs that differ.
    """
    same = pd.Series(True, index=pairs.index)
    for c in DUP_COLS:
        a, b = pairs[c], pairs[f"{c}_kept"]
        same &= (a == b) | (a.isna() & b.isna())

    summary = metric_table({
        "duplicates": len(pairs),
        "same_wine": int(same.sum()),
        "different_wine": int((~same).sum()),
        "pct_same_wine": round(same.mean() * 100, 2) if len(pairs) else 0.0,
    })
    different = pairs.loc[~same, ["wine_id", "title", "title_kept", "winery", "winery_kept"]]
    return summary, different


def metric_table(rows):
    """Turn {metric: value} into a two-column table for a results CSV.
    dtype=object keeps whole numbers as integers in the CSV."""
    return pd.DataFrame({"metric": list(rows.keys()),
                         "value": pd.Series(list(rows.values()), dtype=object)})


def tail_mask(df, unit, max_reviews):
    """True for wines whose `unit` (e.g. winery) has at most `max_reviews`
    reviews in df. Pass the catalogue the counts should come from: from S1
    on that is the catalogue minus test wines (CLAUDE.md rule 4)."""
    counts = df.groupby(unit)[unit].transform("size")
    return counts <= max_reviews


def variety_band(df, edges):
    """Label each wine by how many reviews its variety has.
    With edges [10, 100, 1000]: '1-10', '11-100', '101-1000', '>1000'."""
    counts = df.groupby("variety")["wine_id"].transform("size")
    bins = [0, *edges, np.inf]
    labels = [f"{lo + 1}-{hi}" for lo, hi in zip(bins[:-2], bins[1:-1])] + [f">{edges[-1]}"]
    return pd.cut(counts, bins=bins, labels=labels).astype(str)


def allocate(band_sizes, total, minimum):
    """How many wines to sample from each band.

    Every band gets `minimum` (or all its wines if it has fewer); the rest is
    shared in proportion to band size (largest-remainder rounding). The
    quotas always add up to exactly `total`; impossible settings raise an error.
    """
    sizes = pd.Series(band_sizes, dtype=float)
    if sizes.sum() < total:
        raise ValueError(f"only {int(sizes.sum())} eligible wines, need {total}")
    base = np.minimum(sizes, minimum)
    if base.sum() > total:
        raise ValueError(f"band minimums need {int(base.sum())} wines, more than the total {total}")
    left = total - base.sum()
    if left == 0:
        return base.astype(int)
    room = sizes - base
    share = room / room.sum() * left
    extra = np.floor(share)
    # hand out the rounding leftovers to the largest remainders
    remainder = int(left - extra.sum())
    for band in (share - extra).sort_values(ascending=False).index[:remainder]:
        extra[band] += 1
    return (base + extra).astype(int)


def stratified_sample(df, total, minimum, seed):
    """Sample `total` wines from df, stratified by the 'band' column."""
    quota = allocate(df["band"].value_counts(), total, minimum)
    parts = [df[df["band"] == band].sample(n=n, random_state=seed)
             for band, n in quota.items() if n > 0]
    return pd.concat(parts)


def make_splits(df, cfg, seed):
    """Pick test wines, then dev wines from what is left. Both only from
    notes with at least `min_words` words; the two sets never overlap."""
    s = cfg["split"]
    df = df.assign(band=variety_band(df, s["band_edges"]))
    eligible = df[df["n_words"] >= s["min_words"]]
    log.info("eligible for test/dev (>= %d words): %d", s["min_words"], len(eligible))

    test = stratified_sample(eligible, s["test_size"], s["test_min_per_band"], seed)
    rest = eligible.drop(test.index)
    dev = stratified_sample(rest, s["dev_size"], s["dev_min_per_band"], seed)

    cols = ["wine_id", "variety", "band"]
    return test[cols].sort_values("wine_id"), dev[cols].sort_values("wine_id")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    raw_path = repo_path(cfg["paths"]["raw_csv"])
    out_dir = repo_path(cfg["paths"]["processed_dir"])
    eda_dir = repo_path(cfg["paths"]["eda_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    eda_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(raw_path)
    wines, steps, missing_df, dup_pairs = clean(raw)
    dup_summary, dup_different = duplicate_check(dup_pairs)
    test, dev = make_splits(wines, cfg, cfg["seed"])

    wines.to_parquet(out_dir / "wines.parquet", index=False)
    test.to_csv(out_dir / "test_ids.csv", index=False)
    dev.to_csv(out_dir / "dev_ids.csv", index=False)
    steps.to_csv(eda_dir / "cleaning_log.csv", index=False)
    missing_df.to_csv(eda_dir / "missingness.csv", index=False)
    dup_summary.to_csv(eda_dir / "duplicate_check.csv", index=False)
    dup_different.to_csv(eda_dir / "duplicate_different_wines.csv", index=False)

    manifest = {
        "raw_file": raw_path.name,
        "raw_sha256": file_sha256(raw_path),
        "raw_rows": len(raw),
        "clean_rows": len(wines),
        "test_by_band": test["band"].value_counts().sort_index().to_dict(),
        "dev_by_band": dev["band"].value_counts().sort_index().to_dict(),
        "seed": cfg["seed"],
    }
    with open(eda_dir / "data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("test by band: %s", manifest["test_by_band"])
    log.info("dev by band:  %s", manifest["dev_by_band"])


if __name__ == "__main__":
    main()
