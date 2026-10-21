"""Demo for the supervisor: run the casual queries in data/demo_queries.txt
through Popularity and Semantic and show the top 5 of each.

Run with `make demo` (needs `make data` once; the first run also builds the
embedding cache, which takes a few minutes).

This is a qualitative look at the models, not an evaluation: no metrics are
computed here (those come with the casual test queries in S2).

Outputs
- printed to the terminal
- results/demo/demo_top5.csv   every recommendation, one row each
- results/demo/demo_top5.md    the same, readable, to show on screen
"""
import pandas as pd

from sensoryrec.config import load_config, repo_path
from sensoryrec.recommenders.popularity import PopularityRecommender
from sensoryrec.recommenders.semantic import from_config

SHOW = ["rank", "wine_id", "title", "variety", "country", "price", "points", "score", "description"]


def read_queries(path):
    """One query per line; blank lines and lines starting with # are skipped."""
    lines = [line.strip() for line in open(path, encoding="utf-8")]
    return [line for line in lines if line and not line.startswith("#")]


def short(text, n_words=25):
    words = text.split()
    return " ".join(words[:n_words]) + (" …" if len(words) > n_words else "")


def to_markdown(results):
    parts = ["# Demo: top-5 per model\n",
             "Qualitative output of `make demo`. Not an evaluation.\n"]
    for (i, query), block in results.groupby(["query_no", "query"], sort=True):
        parts.append(f"\n## Query {i}: \"{query}\"\n")
        for model, rows in block.groupby("model", sort=False):
            parts.append(f"\n**{model}**\n")
            parts.append("| # | Wine | Variety | Country | Price | Points | Tasting note (start) |")
            parts.append("|---|---|---|---|---|---|---|")
            for r in rows.itertuples():
                price = "–" if pd.isna(r.price) else f"${r.price:.0f}"
                parts.append(f"| {r.rank} | {r.title} | {r.variety} | {r.country} | {price} "
                             f"| {r.points} | {short(r.description)} |")
    return "\n".join(parts) + "\n"


def main():
    cfg = load_config()
    queries = read_queries(repo_path(cfg["demo"]["queries"]))
    if not queries:
        raise SystemExit(f"No queries yet: add them to {cfg['demo']['queries']}, one per line.")

    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")["wine_id"]

    models = [PopularityRecommender().fit(wines, holdout_ids=test_ids),
              from_config(cfg).fit(wines, holdout_ids=test_ids)]

    blocks = []
    for i, query in enumerate(queries, start=1):
        print(f"\n=== Query {i}: {query}")
        for model in models:
            top = model.recommend(query, k=cfg["k"])[SHOW]
            print(f"\n  {model.name}")
            for r in top.itertuples():
                print(f"   {r.rank}. {r.title}  [{r.variety}, {r.country}, {r.points} pts]")
            blocks.append(top.assign(query_no=i, query=query, model=model.name))

    results = pd.concat(blocks, ignore_index=True)
    out_dir = repo_path(cfg["demo"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "demo_top5.csv", index=False)
    (out_dir / "demo_top5.md").write_text(to_markdown(results), encoding="utf-8")
    print(f"\nSaved to {out_dir}/demo_top5.csv and demo_top5.md")


if __name__ == "__main__":
    main()
