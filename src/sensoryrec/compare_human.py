"""`make compare-human`: LLM-written vs hand-written queries on the same
test wines (DECISIONS.md D1). A validity check, not a significance test:
with about 30 queries, report counts, not p-values.

Inputs
- data/human_queries.csv          the filled-in template (protocol in
                                  docs/human_query_protocol.md); only rows
                                  with a query and consent "yes" are used
- data/queries/test_queries.csv   the LLM queries (flagged ones left out)

Outputs
- results/llm_vs_human.csv          per model and source: queries, hits, HR@k
- results/human_queries_quality.csv the same checks as the LLM queries
  (expert terms, leaked names, length, copy share) for the human queries
  and for the LLM queries of the same wines. Human queries are kept as
  written: no retries.
"""
import logging

import pandas as pd

from sensoryrec import metrics
from sensoryrec.config import load_config, repo_path
from sensoryrec.queries import problems, quality
from sensoryrec.vocab import load_vocab

log = logging.getLogger(__name__)


def load_human(path):
    """Hand-written queries with consent; stops with a message if there are none yet."""
    if not path.exists():
        raise SystemExit(f"No hand-written queries yet: fill in {path} "
                         "(see docs/human_query_protocol.md). Nothing was compared.")
    human = pd.read_csv(path)
    has_query = human["query"].fillna("").astype(str).str.strip() != ""
    consented = human["consent"].fillna("").astype(str).str.strip().str.lower() == "yes"
    return human[has_query & consented].reset_index(drop=True)


def check(queries, wines, vocab):
    """Add the vocabulary and leak checks (one attempt, no retries)."""
    found = [problems(q, wines.loc[w], vocab) for w, q in zip(queries["wine_id"], queries["query"])]
    return queries.assign(first_expert=["; ".join(e) for e, _, _ in found],
                          first_leak=["; ".join(l + o) for _, l, o in found])


def compare(models, sources, k):
    """HR@k of every model on every source of queries (each a DataFrame with
    wine_id and query). Returns one row per model and source."""
    rows = []
    for model in models:
        for name, queries in sources.items():
            recs = [model.recommend(q, k=k)["wine_id"].tolist() for q in queries["query"]]
            targets = queries["wine_id"].tolist()
            hits = sum(t in r[:k] for r, t in zip(recs, targets))
            rows.append({"model": model.name, "source": name, "n_queries": len(queries),
                         "hits": hits, f"hr@{k}": metrics.hit_rate_at_k(recs, targets, k)})
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from sensoryrec.evaluate import load_queries
    from sensoryrec.recommenders import attribute, semantic, tfidf
    from sensoryrec.recommenders.popularity import PopularityRecommender

    cfg = load_config()
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")
    vocab = load_vocab(repo_path(cfg["queries"]["vocab"]))

    human = load_human(repo_path(cfg["human"]["queries"]))
    llm = load_queries(cfg, "test", test_ids)
    shared = set(human["wine_id"]) & set(llm["wine_id"])
    human, llm = human[human["wine_id"].isin(shared)], llm[llm["wine_id"].isin(shared)]
    log.info("%d wines with both an LLM and a human query (%d human queries)", len(shared), len(human))

    by_id = wines.set_index("wine_id", drop=False)
    notes = by_id["description"].to_dict()
    pd.DataFrame([quality(check(human, by_id, vocab), notes, "human"),
                  quality(llm, notes, "llm_same_wines")]).round(3) \
        .to_csv(repo_path("results/human_queries_quality.csv"), index=False)

    models = [PopularityRecommender(), tfidf.from_config(cfg), attribute.from_config(cfg), semantic.from_config(cfg)]
    for model in models:
        model.fit(wines, holdout_ids=test_ids["wine_id"])
    result = compare(models, {"llm": llm, "human": human}, cfg["k"])
    result.round(4).to_csv(repo_path("results/llm_vs_human.csv"), index=False)
    log.info("\n%s", result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
