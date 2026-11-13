"""`make simulate`: about 1,000 simulated customer queries for the coverage
and diversity study (DECISIONS.md, 12 Nov 2026).

Asking an LLM for "1,000 different queries" at temperature 0 gives repeats,
so the variety is built first, without the LLM:
1. `build_briefs` combines the facets of data/simulation_facets.yaml (kind of
   wine, sweetness, flavours, occasion, budget, ...) into distinct "briefs"
   with the fixed seed.
2. The LLM (Gemini, cached) only phrases each brief as one or two everyday
   sentences. The expert-word check applies: on a violation the retry prompt
   names the words, and a query that still fails is flagged and left out.

These queries have no correct wine. They are used by `make eval-simulated`
to measure how many different wines each model surfaces, not accuracy.

Outputs
- data/queries/simulated_queries.csv      query, checks and the brief behind it
- results/simulated_queries_quality.csv   violation, retry and flag rates, length
- data/llm_cache/                         every LLM answer (commit it)
"""
import argparse
import logging

import numpy as np
import pandas as pd
import yaml

from sensoryrec import llm as llm_module
from sensoryrec.config import load_config, repo_path
from sensoryrec.queries import ask_until_clean
from sensoryrec.vocab import expert_terms_used, load_vocab

log = logging.getLogger(__name__)


def build_brief(facets, rng):
    """One brief: who is asking, the kind of wine, and some optional wishes."""
    brief = {"experience": str(rng.choice(facets["experience"])), "kind": str(rng.choice(facets["kind"]))}
    for name, facet in facets["optional"].items():
        if name == "sweetness" and "sweet" in brief["kind"]:
            continue   # "a sweet wine" that is "not sweet at all" would contradict itself
        if rng.random() < facet["chance"]:
            n_pick = int(rng.integers(1, facet.get("pick", 1) + 1))
            picked = rng.choice(facet["values"], size=n_pick, replace=False)
            brief[name] = " and ".join(str(v) for v in picked)
    return brief


def build_briefs(facets, n, seed):
    """n different briefs, the same ones for the same seed."""
    rng = np.random.default_rng(seed)
    briefs, seen = [], set()
    for _ in range(n * 100):   # far more draws than needed; stops as soon as n are found
        brief = build_brief(facets, rng)
        key = tuple(sorted(brief.items()))
        if key not in seen:
            seen.add(key)
            briefs.append(brief)
        if len(briefs) == n:
            return briefs
    raise ValueError(f"the facets only allow {len(briefs)} different briefs, fewer than {n}")


def wishes_text(brief):
    """The brief as the list of wishes shown to the LLM."""
    return "\n".join(f"- {name}: {value}" for name, value in brief.items() if name != "experience")


def quality(queries):
    """One row of quality statistics for the simulated queries."""
    words = queries["query"].str.split().str.len()
    return {"n_queries": len(queries),
            "n_distinct_queries": queries["query"].str.lower().nunique(),
            "pct_expert_first": (queries["first_expert"].fillna("") != "").mean() * 100,
            "pct_regenerated": (queries["attempts"] > 1).mean() * 100,
            "mean_attempts": queries["attempts"].mean(),
            "pct_flagged": queries["flagged"].mean() * 100,
            "words_mean": words.mean(), "words_median": words.median(),
            "words_min": words.min(), "words_max": words.max()}


def check(text, vocab):
    """Problems of one simulated query: expert words, or no text at all."""
    return expert_terms_used(text, vocab) if text else ["an empty answer"]


def simulate(briefs, llm, vocab, prompt, retry_prompt, max_retries):
    """Phrase every brief as a query; returns one row per brief."""
    rows = []
    for i, brief in enumerate(briefs, start=1):
        base = prompt.format(experience=brief["experience"], wishes=wishes_text(brief),
                             banned=", ".join(vocab["expert"]))
        query, attempts, first, remaining = ask_until_clean(
            llm, base, retry_prompt, lambda text: check(text, vocab), max_retries)
        rows.append({"query_id": i, "query": query, "attempts": attempts,
                     "first_expert": "; ".join(first), "violations": "; ".join(remaining),
                     "flagged": bool(remaining), **brief})
        if i % 50 == 0:
            log.info("%d / %d queries (%d new LLM calls)", i, len(briefs), llm.n_calls)
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Write simulated customer queries with the LLM")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N briefs: a cheap trial run before all of them")
    limit = parser.parse_args().limit

    cfg = load_config()
    s = cfg["simulation"]
    with open(repo_path(s["facets"]), encoding="utf-8") as f:
        facets = yaml.safe_load(f)
    vocab = load_vocab(repo_path(cfg["queries"]["vocab"]))
    llm = llm_module.gemini_from_config(cfg)

    briefs = build_briefs(facets, s["n_queries"], cfg["seed"])[:limit]
    queries = simulate(briefs, llm, vocab, s["prompt"], s["retry_prompt"], s["max_retries"])

    out = repo_path(s["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    queries.to_csv(out, index=False)
    summary = pd.DataFrame([quality(queries)]).round(3)
    summary.to_csv(repo_path(s["quality"]), index=False)
    log.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
