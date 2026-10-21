"""`make queries`: the LLM writes one casual, novice-style query per test and
dev wine (DECISIONS.md D1, D2, D7).

For each wine:
1. The fixed prompt (config.yaml) shows the tasting note and asks for a
   1-2 sentence description of the taste in plain words: no expert terms,
   no grape / winery / region / country, no food, occasion or price.
2. The answer is checked: banned expert terms (data/wine_vocab.txt), the
   wine's own names (leak check) and numbers (a price slipping in).
3. On a problem, the retry prompt names the words that were not allowed and
   asks for a rewrite, up to `max_retries` times (stopping early if the model
   repeats itself). If it still fails, the query is kept but flagged, and
   `make eval` leaves it out.

Outputs
- data/queries/{test,dev}_queries.csv   wine_id, query, attempts, checks
- results/queries_quality.csv           violation and retry rates, length,
                                        and how much queries copy their note
- data/llm_cache/                       every LLM answer (commit it)
"""
import argparse
import logging
import re

import numpy as np
import pandas as pd

from sensoryrec import llm as llm_module
from sensoryrec.config import load_config, repo_path
from sensoryrec.text import overlap_share
from sensoryrec.vocab import expert_terms_used, leaked_names, load_vocab

log = logging.getLogger(__name__)


def clean(text):
    """One line, no surrounding quotes or spaces."""
    text = " ".join(str(text).split())
    return text.strip().strip('"“”').strip()


def problems(query, wine, vocab):
    """(expert terms, leaked names, other problems) found in a query."""
    other = ["number or price"] if re.search(r"[\d$€£]", query) else []
    return expert_terms_used(query, vocab), leaked_names(query, wine, vocab["everyday"]), other


def write_query(wine, llm, vocab, prompt, retry_prompt, max_retries):
    """Ask for a query, check it, retry with feedback if needed."""
    base = prompt.format(note=wine["description"], banned=", ".join(vocab["expert"]))
    query = clean(llm.generate(base))
    expert, leak, other = problems(query, wine, vocab)
    first = {"first_expert": "; ".join(expert), "first_leak": "; ".join(leak + other)}
    attempts, retries = 1, 0   # attempts = different answers; retries = retry prompts sent
    while (expert or leak or other) and retries < max_retries:
        found = ", ".join(expert + leak + other)
        new = clean(llm.generate(base + retry_prompt.format(previous=query, problems=found)))
        retries += 1
        if new == query:
            # Same answer again: the next retry prompt would be identical, and
            # with temperature 0 so would the answer. Stop trying.
            break
        query, attempts = new, attempts + 1
        expert, leak, other = problems(query, wine, vocab)
    remaining = expert + leak + other
    return {"wine_id": wine["wine_id"], "query": query, "attempts": attempts, **first,
            "violations": "; ".join(remaining), "flagged": bool(remaining)}


def non_empty(queries, col):
    """True where the column has text (False everywhere if it is missing)."""
    if col not in queries:
        return pd.Series(False, index=queries.index)
    return queries[col].fillna("").astype(str) != ""


def quality(queries, notes, label):
    """One row of quality statistics for a set of queries.
    notes: wine_id -> tasting note, for the copy measure."""
    words = queries["query"].str.split().str.len()
    copy = [overlap_share(q, notes[w]) for w, q in zip(queries["wine_id"], queries["query"])]
    expert, leak = non_empty(queries, "first_expert"), non_empty(queries, "first_leak")
    row = {"set": label, "n_queries": len(queries),
           "pct_expert_first": expert.mean() * 100,
           "pct_leak_first": leak.mean() * 100,
           "pct_any_first": (expert | leak).mean() * 100}
    if "attempts" in queries:
        row.update({"pct_regenerated": (queries["attempts"] > 1).mean() * 100,
                    "mean_attempts": queries["attempts"].mean(),
                    "pct_flagged": queries["flagged"].mean() * 100})
    row.update({"words_mean": words.mean(), "words_median": words.median(),
                "words_min": words.min(), "words_max": words.max(),
                "copy_share_mean": float(np.mean(copy)), "copy_share_median": float(np.median(copy))})
    return row


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Write casual queries with the local LLM")
    parser.add_argument("--split", choices=["test", "dev", "all"], default="all")
    split = parser.parse_args().split

    cfg = load_config()
    q = cfg["queries"]
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet").set_index("wine_id", drop=False)
    vocab = load_vocab(repo_path(q["vocab"]))
    llm = llm_module.from_config(cfg)
    out_dir = repo_path(q["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in (["test", "dev"] if split == "all" else [split]):
        ids = pd.read_csv(processed / f"{name}_ids.csv")["wine_id"].sort_values()
        rows = []
        for i, wine_id in enumerate(ids, start=1):
            rows.append(write_query(wines.loc[wine_id], llm, vocab, q["prompt"],
                                    q["retry_prompt"], q["max_retries"]))
            if i % 20 == 0:
                log.info("%s: %d / %d queries (%d new LLM calls)", name, i, len(ids), llm.n_calls)
        pd.DataFrame(rows).to_csv(out_dir / f"{name}_queries.csv", index=False)

    notes = wines["description"].to_dict()
    summary = [quality(pd.read_csv(out_dir / f"{name}_queries.csv"), notes, f"llm_{name}")
               for name in ["test", "dev"] if (out_dir / f"{name}_queries.csv").exists()]
    path = repo_path(q["quality"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).round(3).to_csv(path, index=False)
    log.info("\n%s", pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
