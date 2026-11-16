"""`make judge`: an LLM rates how good each model's top 5 is for a request
(LLM-as-a-judge; DECISIONS.md, 16 Nov 2026; method after Zhen's thesis,
section 3.6).

Why: the exact-wine test (S2) only rewards THE wine a query was written
from; a different but equally good wine counts as a miss. A judge rates
every recommended wine against the request instead.

1. Tiers. The simulated requests keep the wishes they were built from, so
   `tier` sorts them without new queries: basic (1-2 wishes, no occasion),
   medium (3-4 wishes, no occasion), complex (an occasion + at least 2 other
   wishes). `sample_tiers` draws 50 per tier with the seed.
2. Lists. Every model gives its top k for each request; a random baseline
   (wines drawn with the seed) shows what "no understanding" scores.
3. Rating. The judge (Gemini, temperature 0, cached, pinned) sees the
   request and the k wines (tasting note, grape, country, price; never the
   model's name) and returns JSON: for each wine a short reason and a 1-5
   relevance score, then a 1-5 score for the whole set, and for complex
   requests a 1-5 score for how well the set fits the occasion. The rubric
   is in config.yaml.
4. Head to head. For the two semantic models, the judge sees both lists (in
   a seeded random order, as "A" and "B") and picks the better one or a tie.

The judge never ranks the recommendations used by the system (CLAUDE.md
rule 2): it only scores lists that the models have already produced.

Outputs (results/judge/)
- sample.csv          the 150 requests and their tier
- items.csv           one row per request x model x wine: relevance 1-5
- sets.csv            one row per request x model: set relevance, occasion fit
- pairwise.csv        one row per request: winner of the two semantic models
- summary.csv         mean scores and share of relevant wines (>= 4) per model and tier
- pairwise_summary.csv  win / tie rates per tier
- validation_sample.csv 30 sets to score by hand (`make judge-agree` compares)
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

from sensoryrec import llm as llm_module
from sensoryrec.config import load_config, repo_path

log = logging.getLogger(__name__)

WISH_COLUMNS = ["sweetness", "weight", "flavours", "feel", "occasion", "budget", "country", "also"]
TIERS = ["basic", "medium", "complex"]


def tier(row):
    """basic / medium / complex, or None if the request fits no tier
    (DECISIONS.md, 16 Nov 2026). A wish is any filled wish column; the kind
    of wine is always there and does not count."""
    wishes = [c for c in WISH_COLUMNS if c in row and isinstance(row[c], str) and row[c].strip()]
    has_occasion = "occasion" in wishes
    others = len(wishes) - has_occasion
    if has_occasion:
        return "complex" if others >= 2 else None
    if 1 <= len(wishes) <= 2:
        return "basic"
    if 3 <= len(wishes) <= 4:
        return "medium"
    return None


def sample_tiers(queries, n_per_tier, seed):
    """n requests per tier, drawn with the seed from the usable (not flagged)
    simulated requests; sorted by tier then query_id."""
    usable = queries[~queries["flagged"].astype(bool)].copy()
    usable["tier"] = usable.apply(tier, axis=1)
    parts = []
    for i, t in enumerate(TIERS):
        pool = usable[usable["tier"] == t]
        if len(pool) < n_per_tier:
            raise ValueError(f"only {len(pool)} {t} requests, fewer than {n_per_tier}")
        parts.append(pool.sample(n=n_per_tier, random_state=seed + i))
    out = pd.concat(parts)
    out["tier"] = pd.Categorical(out["tier"], TIERS, ordered=True)
    return out.sort_values(["tier", "query_id"]).reset_index(drop=True)


def random_lists(query_ids, wine_ids, k, seed):
    """The random baseline: k different wines per request, drawn with the seed."""
    rng = np.random.default_rng(seed)
    wine_ids = np.asarray(wine_ids)
    return {q: rng.choice(wine_ids, size=k, replace=False).tolist() for q in query_ids}


def describe(wines, wine_ids):
    """The wines as the judge sees them: numbered, no name, no model."""
    lines = []
    for i, w in enumerate(wine_ids, start=1):
        r = wines.loc[w]
        price = f"${r['price']:.0f}" if pd.notna(r["price"]) else "price unknown"
        lines.append(f"Wine {i}: {r['variety']}, {r['country']}, {price}. Tasting note: \"{r['description']}\"")
    return "\n".join(lines)


def parse_scores(text, k, with_occasion):
    """Read the judge's JSON answer. Returns (item scores, set score,
    occasion score or None); raises ValueError if anything is missing or out
    of range, so a broken answer is never counted silently."""
    data = json.loads(text)
    items = data["wines"]
    if len(items) != k:
        raise ValueError(f"expected {k} wines, got {len(items)}")
    scores = [int(item["relevance"]) for item in items]
    set_score = int(data["set_relevance"])
    occasion = int(data["occasion_fit"]) if with_occasion else None
    for s in scores + [set_score] + ([occasion] if with_occasion else []):
        if not 1 <= s <= 5:
            raise ValueError(f"score {s} outside 1-5")
    return scores, set_score, occasion


def parse_pairwise(text):
    """'A', 'B' or 'tie' from the head-to-head answer."""
    winner = str(json.loads(text)["better"]).strip()
    if winner not in {"A", "B", "tie"}:
        raise ValueError(f"unexpected winner {winner!r}")
    return winner


def judge_sets(sample, lists, wines, llm, prompt, k):
    """Rate every (request, model) list. lists: model -> {query_id: wine_ids}.
    Goes request by request through all models, so a run stopped by the
    quota still leaves complete comparisons for the requests it reached."""
    items, sets = [], []
    for row in sample.itertuples():
        with_occasion = row.tier == "complex"
        for model, by_query in lists.items():
            wine_ids = by_query[row.query_id]
            text = llm.generate(prompt.format(
                request=row.query, wines=describe(wines, wine_ids), k=k,
                occasion_rule=OCCASION_RULE if with_occasion else "",
                occasion_field=OCCASION_FIELD if with_occasion else ""))
            try:
                scores, set_score, occasion = parse_scores(text, k, with_occasion)
            except (ValueError, KeyError, TypeError) as error:
                log.warning("unusable answer for request %s / %s: %s", row.query_id, model, error)
                sets.append({"query_id": row.query_id, "tier": row.tier, "model": model,
                             "set_relevance": None, "occasion_fit": None, "parsed": False})
                continue
            items += [{"query_id": row.query_id, "tier": row.tier, "model": model, "rank": r,
                       "wine_id": w, "relevance": s}
                      for r, (w, s) in enumerate(zip(wine_ids, scores), start=1)]
            sets.append({"query_id": row.query_id, "tier": row.tier, "model": model,
                         "set_relevance": set_score, "occasion_fit": occasion, "parsed": True})
    return pd.DataFrame(items), pd.DataFrame(sets)


OCCASION_RULE = "- occasion_fit: how well the wines suit the occasion in the request (1-5).\n"
OCCASION_FIELD = ', "occasion_fit": <1-5>'


def judge_pairwise(sample, list_a, list_b, names, wines, llm, prompt, seed):
    """Head to head between two models; the order shown to the judge is
    random (seeded) per request, so position cannot favour one model."""
    rng = np.random.default_rng(seed)
    rows = []
    for row in sample.itertuples():
        swap = bool(rng.integers(2))
        first, second = (list_b, list_a) if swap else (list_a, list_b)
        text = llm.generate(prompt.format(request=row.query,
                                          list_a=describe(wines, first[row.query_id]),
                                          list_b=describe(wines, second[row.query_id])))
        try:
            shown = parse_pairwise(text)
        except (ValueError, KeyError, TypeError) as error:
            log.warning("unusable head-to-head answer for request %s: %s", row.query_id, error)
            shown = None
        winner = None
        if shown == "tie":
            winner = "tie"
        elif shown is not None:
            winner = names[1] if (shown == "A") == swap else names[0]
        rows.append({"query_id": row.query_id, "tier": row.tier, "shown_first": names[1] if swap else names[0],
                     "answer": shown, "winner": winner})
    return pd.DataFrame(rows)


def summarise(items, sets):
    """Mean scores per model and tier (and over all tiers), plus the share of
    wines rated relevant (4 or 5)."""
    def block(group_cols):
        it = items.groupby(group_cols).agg(item_relevance=("relevance", "mean"),
                                          pct_relevant=("relevance", lambda s: (s >= 4).mean() * 100))
        st = sets[sets["parsed"]].groupby(group_cols).agg(
            set_relevance=("set_relevance", "mean"), occasion_fit=("occasion_fit", "mean"),
            n_sets=("set_relevance", "size"))
        return it.join(st).reset_index()
    by_tier = block(["model", "tier"])
    overall = block(["model"]).assign(tier="all")
    return pd.concat([by_tier, overall], ignore_index=True)


def pairwise_summary(pairwise, names):
    rows = []
    for t, g in list(pairwise.groupby("tier", observed=True)) + [("all", pairwise)]:
        g = g.dropna(subset=["winner"])
        if g.empty:
            continue
        rows.append({"tier": t, "n": len(g),
                     f"pct_{names[0]}_wins": (g["winner"] == names[0]).mean() * 100,
                     f"pct_{names[1]}_wins": (g["winner"] == names[1]).mean() * 100,
                     "pct_tie": (g["winner"] == "tie").mean() * 100})
    return pd.DataFrame(rows)


def validation_sample(sets, sample, lists, wines, n, seed):
    """n judged sets (all models mixed, model hidden) for the student to
    score by hand with the same rubric; `agreement` compares afterwards."""
    done = sets[sets["parsed"]].sample(n=min(n, int(sets["parsed"].sum())), random_state=seed)
    query = sample.set_index("query_id")["query"]
    rows = []
    for r in done.itertuples():
        rows.append({"case": len(rows) + 1, "query_id": r.query_id, "model_hidden": r.model,
                     "request": query[r.query_id],
                     "wines": describe(wines, lists[r.model][r.query_id]),
                     "your_set_relevance_1_to_5": ""})
    return pd.DataFrame(rows)


def agreement(human, judge):
    """How often the student and the judge agree on set relevance:
    exact match, within one point, and Spearman correlation."""
    m = human.merge(judge, on=["query_id", "model"])
    if m.empty:
        raise ValueError("no case was scored by both")
    diff = (m["human"] - m["judge"]).abs()
    return {"n": len(m), "pct_exact": (diff == 0).mean() * 100,
            "pct_within_1": (diff <= 1).mean() * 100,
            "spearman": float(m["human"].corr(m["judge"], method="spearman"))}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="LLM-as-a-judge on the simulated requests")
    parser.add_argument("--agree", action="store_true", help="compare the hand-scored validation sample")
    args = parser.parse_args()

    cfg = load_config()
    j = cfg["judge"]
    out = repo_path(j["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if args.agree:
        filled = pd.read_csv(repo_path(j["validation_filled"]))
        filled = filled[pd.to_numeric(filled["your_set_relevance_1_to_5"], errors="coerce").notna()]
        human = filled.rename(columns={"model_hidden": "model", "your_set_relevance_1_to_5": "human"})
        human["human"] = human["human"].astype(int)
        judge = pd.read_csv(out / "sets.csv").rename(columns={"set_relevance": "judge"})
        result = agreement(human[["query_id", "model", "human"]], judge[["query_id", "model", "judge"]])
        pd.DataFrame([result]).round(3).to_csv(out / "agreement.csv", index=False)
        log.info("%s", result)
        return

    from sensoryrec.recommenders import build_models
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")["wine_id"]
    queries = pd.read_csv(repo_path(cfg["simulation"]["out"]))
    sample = sample_tiers(queries, j["n_per_tier"], cfg["seed"])
    sample[["query_id", "tier", "query"]].to_csv(out / "sample.csv", index=False)

    k = j["k"]
    lists = {}
    for model in build_models(cfg):
        if model.name not in j["models"]:
            continue
        model.fit(wines, holdout_ids=test_ids)
        lists[model.name] = {q: model.recommend(text, k=k)["wine_id"].tolist()
                             for q, text in zip(sample["query_id"], sample["query"])}
        log.info("%s: lists ready", model.name)
    lists["random"] = random_lists(sample["query_id"], wines["wine_id"], k, cfg["seed"])

    by_id = wines.set_index("wine_id")
    llm = llm_module.gemini_from_config(cfg, section="judge")
    items, sets = judge_sets(sample, lists, by_id, llm, j["prompt"], k)
    items.to_csv(out / "items.csv", index=False)
    sets.to_csv(out / "sets.csv", index=False)
    summarise(items, sets).round(3).to_csv(out / "summary.csv", index=False)
    log.info("%d / %d answers usable", int(sets["parsed"].sum()), len(sets))

    names = j["pairwise"]
    pw = judge_pairwise(sample, lists[names[0]], lists[names[1]], names, by_id, llm,
                        j["pairwise_prompt"], cfg["seed"])
    pw.to_csv(out / "pairwise.csv", index=False)
    pairwise_summary(pw, names).round(2).to_csv(out / "pairwise_summary.csv", index=False)

    validation_sample(sets, sample, lists, by_id, j["n_validation"], cfg["seed"]) \
        .to_csv(out / "validation_sample.csv", index=False)
    log.info("\n%s", summarise(items, sets).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
