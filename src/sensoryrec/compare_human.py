"""`make compare-human`: LLM-written vs hand-written queries on the same
test wines (DECISIONS.md D1). A validity check, not a significance test:
with about 30 queries, report counts, not p-values.

Inputs
- data/human_queries.csv          the filled-in template (protocol in
                                  docs/human_query_protocol.md); only rows
                                  with a query and consent "yes" are used
- data/queries/test_queries.csv   the LLM queries (flagged ones left out)

Outputs
- results/llm_vs_human.csv          per model and source: queries, hits, HR@k.
  Sources, all on the same wines (those with a usable LLM query):
  "llm" (after retries, flagged ones left out), "human_all" (as written)
  and "human_clean" (human queries that pass the same checks).
- results/human_queries_quality.csv the same checks for the human queries
  and for ALL LLM queries of the same wines (flagged ones included), so the
  two rows are comparable; plus how many wines were lost because their LLM
  query was flagged. Human queries are kept as written: no retries.
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


def prepare(human, llm_all, wines_by_id, vocab):
    """Line up human and LLM queries on the same wines.

    Returns (sources for the HR comparison, all LLM queries of the human
    wines, checked human queries, wines lost because the LLM query was flagged).
    """
    human = check(human, wines_by_id, vocab)
    llm_same = llm_all[llm_all["wine_id"].isin(human["wine_id"])]
    usable = set(llm_same.loc[~llm_same["flagged"].astype(bool), "wine_id"])
    lost = set(human["wine_id"]) - usable
    passes = (human["first_expert"] == "") & (human["first_leak"] == "")
    sources = {"llm": llm_same[llm_same["wine_id"].isin(usable)],
               "human_all": human[human["wine_id"].isin(usable)],
               "human_clean": human[human["wine_id"].isin(usable) & passes]}
    return sources, llm_same, human, lost


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
    from sensoryrec.recommenders import attribute, semantic, tfidf
    from sensoryrec.recommenders.popularity import PopularityRecommender

    cfg = load_config()
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")
    vocab = load_vocab(repo_path(cfg["queries"]["vocab"]))
    by_id = wines.set_index("wine_id", drop=False)

    human = load_human(repo_path(cfg["human"]["queries"]))
    llm_all = pd.read_csv(repo_path(cfg["queries"]["out_dir"]) / "test_queries.csv")  # flagged included
    sources, llm_same, human, lost = prepare(human, llm_all, by_id, vocab)
    log.info("%d human queries; %d wines left out because their LLM query was flagged",
             len(human), len(lost))

    notes = by_id["description"].to_dict()
    human_row = quality(human, notes, "human")
    human_row["n_wines_lost_to_llm_flags"] = len(lost)
    pd.DataFrame([human_row, quality(llm_same, notes, "llm_same_wines")]).round(3) \
        .to_csv(repo_path("results/human_queries_quality.csv"), index=False)

    models = [PopularityRecommender(), tfidf.from_config(cfg), attribute.from_config(cfg), semantic.from_config(cfg)]
    for model in models:
        model.fit(wines, holdout_ids=test_ids["wine_id"])
    result = compare(models, sources, cfg["k"])
    result.round(4).to_csv(repo_path("results/llm_vs_human.csv"), index=False)
    log.info("\n%s", result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
