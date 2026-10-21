"""Plain-language refinement for the Semantic model: "sweeter", "less oaky",
"cheaper" (DECISIONS.md D4, D5).

Taste traits move the query in embedding space:
- Direction vector d for a trait = mean embedding of the training notes that
  contain a *build* word, minus the mean of those that don't, scaled to
  length 1. It points from "notes without the trait" towards "notes with it".
- "more X":  q' = normalise(q + alpha * d);  "less X": q' = normalise(q - alpha * d)
"Cheaper" is not a direction but a price filter: at most the median price of
the current top 5.

Whether a shift worked is judged with a *separate* measure list per trait:
the share of top-5 notes containing a measure word, before vs after. Build
and measure lists share no word stem, so success is not the build words
simply coming back.

`make refine`
1. Chooses alpha on the dev queries: the smallest alpha in the grid whose
   mean success is within `tolerance_pp` points of the best
   -> results/refinement_alpha_dev.csv
2. Evaluates on the test queries with that alpha, with two controls:
   (a) random directions with the same alpha, (b) how far the target wine
   drops -> results/refinement.csv
3. Exports 10 test cases for the sommelier -> results/refinement_sommelier_sample.csv
Test notes and test queries are never used to build vectors or choose alpha.
"""
import logging

import numpy as np
import pandas as pd
import yaml

from sensoryrec.config import load_config, repo_path
from sensoryrec.recommenders.base import rank
from sensoryrec.text import any_term_pattern, normalise

log = logging.getLogger(__name__)


def load_lexicons(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def has_terms(notes, terms, already_normalised=False):
    """True for each note containing at least one of the terms (whole words,
    plural allowed; a term right after "not", "no", ... does not count).
    Pass already_normalised=True to skip normalising the notes again."""
    pattern = any_term_pattern(terms, negation_guard=True)
    if not already_normalised:
        notes = [normalise(n) for n in notes]
    return np.array([bool(pattern.search(n)) for n in notes])


def unit(v):
    return v / np.linalg.norm(v)


def direction_vector(embeddings, has_trait):
    """Mean of the notes with the trait minus mean of those without, length 1."""
    return unit(embeddings[has_trait].mean(axis=0) - embeddings[~has_trait].mean(axis=0))


def shift(query_vec, direction, alpha, sign=1):
    """Move the query along the direction: sign +1 = "more", -1 = "less"."""
    return unit(query_vec + sign * alpha * direction)


def top_k(scores, wine_ids, k):
    """Positions of the k best scores, ties broken by wine_id (as in rank())."""
    threshold = np.partition(scores, -k)[-k]
    candidates = np.flatnonzero(scores >= threshold)
    order = np.lexsort((wine_ids[candidates], -scores[candidates]))
    return candidates[order[:k]]


def target_rank(scores, wine_ids, position):
    """Rank (1 = best) of the wine at `position`, ties broken by wine_id."""
    s, w = scores[position], wine_ids[position]
    return int(1 + (scores > s).sum() + ((scores == s) & (wine_ids < w)).sum())


def is_success(before, after, sign):
    """The measured share moved the requested way (strictly)."""
    return after > before if sign > 0 else after < before


def is_eligible(before, sign):
    """Success is possible: "more" needs room to grow, "less" something to remove."""
    return before < 1 if sign > 0 else before > 0


def choose_alpha(mean_success, tolerance_pp):
    """The smallest alpha whose mean success is within tolerance_pp
    percentage points of the best (mean_success: alpha -> rate in [0, 1])."""
    best = max(mean_success.values())
    return min(a for a, rate in mean_success.items() if rate >= best - tolerance_pp / 100)


class Refiner:
    def __init__(self, embeddings, catalogue, lexicons):
        """embeddings: the Semantic model's note embeddings (unit length), one
        row per catalogue row. catalogue: the wines, same order."""
        self.embeddings = embeddings
        self.catalogue = catalogue.reset_index(drop=True)
        self.wine_ids = self.catalogue["wine_id"].to_numpy()
        self.lexicons = lexicons

    def fit(self, holdout_ids):
        """Direction vectors from training notes only (test wines held out);
        measure masks over the whole catalogue (they judge results)."""
        train = ~self.catalogue["wine_id"].isin(set(holdout_ids)).to_numpy()
        notes = np.array([normalise(n) for n in self.catalogue["description"]])  # once, not per list
        self.directions, self.measure = {}, {}
        for trait, lex in self.lexicons.items():
            has_trait = has_terms(notes[train], lex["build"], already_normalised=True)
            self.directions[trait] = direction_vector(self.embeddings[train], has_trait)
            self.measure[trait] = has_terms(notes, lex["measure"], already_normalised=True)
            log.info("%s: %d training notes with a build word", trait, has_trait.sum())
        return self

    def scores(self, query_vec):
        # same number type as the embeddings, so numpy does not convert all of them
        return self.embeddings @ np.asarray(query_vec, dtype=self.embeddings.dtype)

    def evaluate(self, query_vecs, target_ids, alpha, k=5, random_dirs=None):
        """One row per query x trait x direction: measured share before/after,
        eligible, success, the target's rank before/after, and (if random_dirs
        is given) the success rate of the random directions."""
        position_of = {w: i for i, w in enumerate(self.wine_ids)}
        rows = []
        for q, target in zip(query_vecs, target_ids):
            base = self.scores(q)
            before = top_k(base, self.wine_ids, k)
            rank_before = target_rank(base, self.wine_ids, position_of[target])
            random_tops = {}
            if random_dirs is not None:
                for sign in (1, -1):
                    random_tops[sign] = [top_k(self.scores(shift(q, r, alpha, sign)), self.wine_ids, k)
                                         for r in random_dirs]
            for trait, d in self.directions.items():
                m = self.measure[trait]
                share_before = m[before].mean()
                for sign in (1, -1):
                    moved = self.scores(shift(q, d, alpha, sign))
                    share_after = m[top_k(moved, self.wine_ids, k)].mean()
                    row = {"wine_id": target, "trait": trait, "sign": sign,
                           "share_before": share_before, "share_after": share_after,
                           "eligible": is_eligible(share_before, sign),
                           "success": is_success(share_before, share_after, sign),
                           "rank_before": rank_before,
                           "rank_after": target_rank(moved, self.wine_ids, position_of[target])}
                    if random_dirs is not None:
                        row["random_success"] = np.mean([is_success(share_before, m[t].mean(), sign)
                                                         for t in random_tops[sign]])
                    rows.append(row)
        return pd.DataFrame(rows)

    def cheaper(self, query_vecs, target_ids, k=5):
        """The "cheaper" refinement: max price = median price of the current
        top k (wines without a price excluded). One row per query."""
        rows = []
        for q, target in zip(query_vecs, target_ids):
            s = self.scores(q)
            before = rank(self.catalogue, s, k)
            median_before = before["price"].median()
            if pd.isna(median_before):
                rows.append({"wine_id": target, "eligible": False})
                continue
            after = rank(self.catalogue, s, k, filters={"max_price": median_before})
            rows.append({"wine_id": target, "eligible": True,
                         "success": after["price"].median() < median_before,
                         "target_in_before": target in set(before["wine_id"]),
                         "target_in_after": target in set(after["wine_id"])})
        return pd.DataFrame(rows)


def summarise(cases, lexicons):
    """Per trait and direction: success rate (over eligible cases), the random
    control, and the cost to the target wine."""
    rows = []
    for (trait, sign), g in cases.groupby(["trait", "sign"], sort=False):
        ok = g[g["eligible"]]
        in_top = g[g["rank_before"] <= 5]
        row = {"refinement": lexicons[trait]["more" if sign > 0 else "less"],
               "trait": trait, "direction": "more" if sign > 0 else "less",
               "n_queries": len(g), "n_eligible": len(ok),
               "success_rate": ok["success"].mean() if len(ok) else np.nan}
        if "random_success" in g:
            row["random_success_rate"] = ok["random_success"].mean() if len(ok) else np.nan
        row.update({"median_rank_change": (g["rank_after"] - g["rank_before"]).median(),
                    "n_target_in_top5_before": len(in_top),
                    "pct_target_left_top5": (in_top["rank_after"] > 5).mean() * 100 if len(in_top) else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_cheaper(cases):
    ok = cases[cases["eligible"]]
    in_top = ok[ok["target_in_before"]]
    return {"refinement": "cheaper", "trait": "price", "direction": "less",
            "n_queries": len(cases), "n_eligible": len(ok), "success_rate": ok["success"].mean(),
            "n_target_in_top5_before": len(in_top),
            "pct_target_left_top5": (~in_top["target_in_after"]).mean() * 100 if len(in_top) else np.nan}


def macro_success(cases):
    """Mean of the per trait-and-direction success rates (each of the 12
    refinements counts equally)."""
    ok = cases[cases["eligible"]]
    return ok.groupby(["trait", "sign"])["success"].mean().mean()


def sommelier_sample(cases, queries, catalogue, lexicons, refiner, query_vecs, alpha, n, seed, k=5):
    """n eligible test cases with the query, the refinement and the top-5
    titles and notes before and after, for the sommelier check (D5)."""
    pick = cases[cases["eligible"]].sample(n=min(n, int(cases["eligible"].sum())), random_state=seed)
    vec_of = dict(zip(queries["wine_id"], query_vecs))
    query_of = dict(zip(queries["wine_id"], queries["query"]))

    def describe(positions):
        wines = catalogue.iloc[positions]
        return " || ".join(f"{t}: {d}" for t, d in zip(wines["title"], wines["description"]))

    rows = []
    for c in pick.itertuples():
        q = vec_of[c.wine_id]
        moved = shift(q, refiner.directions[c.trait], alpha, c.sign)
        rows.append({"wine_id": c.wine_id, "query": query_of[c.wine_id],
                     "refinement": lexicons[c.trait]["more" if c.sign > 0 else "less"],
                     "top5_before": describe(top_k(refiner.scores(q), refiner.wine_ids, k)),
                     "top5_after": describe(top_k(refiner.scores(moved), refiner.wine_ids, k)),
                     "sommelier_moved_right_way": "", "sommelier_comment": ""})
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from sensoryrec.evaluate import load_queries
    from sensoryrec.recommenders import semantic

    cfg = load_config()
    r = cfg["refinement"]
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")
    dev = load_queries(cfg, "dev", pd.read_csv(processed / "dev_ids.csv"))
    test = load_queries(cfg, "test", test_ids)

    model = semantic.from_config(cfg).fit(wines, holdout_ids=test_ids["wine_id"])
    lexicons = load_lexicons(repo_path(r["lexicons"]))
    refiner = Refiner(model.embeddings, model.catalogue, lexicons).fit(test_ids["wine_id"])
    dev_vecs, test_vecs = model.encode(dev["query"].tolist()), model.encode(test["query"].tolist())
    k = cfg["k"]

    # 1. alpha on dev only
    grid = {a: macro_success(refiner.evaluate(dev_vecs, dev["wine_id"], a, k)) for a in r["alphas"]}
    alpha = choose_alpha(grid, r["tolerance_pp"])
    out = repo_path(r["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"alpha": list(grid), "mean_success_dev": list(grid.values()),
                  "chosen": [a == alpha for a in grid]}).round(4).to_csv(out / "refinement_alpha_dev.csv", index=False)
    log.info("alpha chosen on dev: %s", alpha)
    if alpha == max(r["alphas"]):
        log.warning("alpha is the largest value in the grid: success may still rise for larger "
                    "alpha; consider extending `refinement.alphas` (decide before looking at test)")

    # 2. test, with controls
    rng = np.random.default_rng(cfg["seed"])
    random_dirs = [unit(v) for v in rng.normal(size=(r["n_random"], model.embeddings.shape[1]))]
    cases = refiner.evaluate(test_vecs, test["wine_id"], alpha, k, random_dirs)
    summary = summarise(cases, lexicons)
    summary = pd.concat([summary, pd.DataFrame([summarise_cheaper(refiner.cheaper(test_vecs, test["wine_id"], k))])],
                        ignore_index=True)
    summary.insert(0, "alpha", alpha)
    summary.round(4).to_csv(out / "refinement.csv", index=False)

    # 3. cases for the sommelier
    sommelier_sample(cases, test, model.catalogue, lexicons, refiner, test_vecs, alpha,
                     r["sommelier_cases"], cfg["seed"], k).to_csv(out / "refinement_sommelier_sample.csv", index=False)
    log.info("\n%s", summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
