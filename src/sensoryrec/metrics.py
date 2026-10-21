"""Evaluation metrics.

Conventions
- `recs` is a list of recommendation lists, one per query, each a list of
  wine_ids, best first.
- `targets` is the one correct wine_id per query (self-retrieval: the wine
  whose tasting note the casual query was written from).

Accuracy:      hit_rate_at_k, ndcg_at_k
Long tail:     catalogue_coverage, tail_share, recommendation_gini, novelty
Diversity:     intra_list_diversity

When there is nothing to measure (no queries, no recommendations, no pairs),
a metric raises ValueError instead of returning nan, so an empty result can
never slip into a results table unnoticed.
"""
from collections import Counter
from itertools import chain

import numpy as np


def gini(counts):
    """Gini coefficient of a list of non-negative counts.

    0 = every item has the same count; close to 1 = one item has everything.
    Formula on counts sorted ascending x_1..x_n (i = 1..n):
        G = 2 * sum(i * x_i) / (n * sum(x)) - (n + 1) / n
    """
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float(2 * np.sum(i * x) / (n * x.sum()) - (n + 1) / n)


def hit_rate_at_k(recs, targets, k):
    """Share of queries whose target wine is in the top k."""
    if not recs:
        raise ValueError("no queries")
    hits = [target in list(rec)[:k] for rec, target in zip(recs, targets)]
    return float(np.mean(hits))


def ndcg_at_k(recs, targets, k):
    """NDCG@k with one relevant wine per query.

    A hit at rank r (1 = first) earns 1 / log2(r + 1); a miss earns 0. With a
    single relevant wine the best possible score is a hit at rank 1, which
    earns exactly 1, so no further normalisation is needed. Averaged over
    queries. Unlike HR@k, it rewards finding the wine higher up the list.
    """
    if not recs:
        raise ValueError("no queries")
    scores = []
    for rec, target in zip(recs, targets):
        top = list(rec)[:k]
        scores.append(1 / np.log2(top.index(target) + 2) if target in top else 0.0)
    return float(np.mean(scores))


def catalogue_coverage(recs, catalogue_size):
    """Share of the catalogue that appears in at least one recommendation list."""
    return len(set(chain.from_iterable(recs))) / catalogue_size


def tail_share(recs, tail_ids):
    """Share of all recommendation slots filled by long-tail wines."""
    tail_ids = set(tail_ids)
    slots = list(chain.from_iterable(recs))
    if not slots:
        raise ValueError("no recommendations")
    return sum(item in tail_ids for item in slots) / len(slots)


def recommendation_gini(recs, all_items):
    """How unequally recommendations are spread over `all_items`.

    Counts how often each item is recommended; items never recommended count
    as 0. 0 = every item recommended equally often; near 1 = a few items get
    everything. For the winery level, pass lists of wineries instead of wine_ids
    and all wineries as `all_items`.
    """
    counts = Counter(chain.from_iterable(recs))
    return gini([counts.get(item, 0) for item in all_items])


def novelty(recs, item_popularity):
    """Average self-information of the recommended wines.

    item_popularity maps wine_id -> p, the share of all reviews that belong
    to the wine's winery (DECISIONS.md). A wine from a rarely reviewed winery
    has a small p, so -log2(p) is large: it is more novel. Averaged within
    each list, then over lists; empty lists are skipped.

    Every p must be > 0: -log2(0) is infinite. A wine whose winery has no
    reviews outside the test set would get p = 0; how to handle that is an
    open decision in DECISIONS.md, so it is an error here, not a silent inf.
    """
    lists = [rec for rec in recs if len(rec) > 0]
    if not lists:
        raise ValueError("no recommendations")
    bad = [item for rec in lists for item in rec if item_popularity[item] <= 0]
    if bad:
        raise ValueError(f"popularity must be > 0 for novelty; p <= 0 for {bad[:5]} "
                         "(see the open item in DECISIONS.md)")
    per_list = [np.mean([-np.log2(item_popularity[item]) for item in rec]) for rec in lists]
    return float(np.mean(per_list))


def intra_list_diversity(recs, vectors):
    """1 - average cosine similarity between every pair of wines in a list,
    averaged over lists. vectors maps wine_id -> embedding. 0 = all wines in
    the list are identical in meaning; higher = more varied. Lists with
    fewer than 2 wines are skipped (there are no pairs).
    """
    values = []
    for rec in recs:
        if len(rec) < 2:
            continue
        m = np.stack([vectors[item] for item in rec]).astype(float)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        sims = m @ m.T
        n = len(rec)
        # sum of all pairs, minus the diagonal (each wine with itself)
        mean_pair_sim = (sims.sum() - np.trace(sims)) / (n * (n - 1))
        values.append(1 - mean_pair_sim)
    if not values:
        raise ValueError("no list has 2 or more wines, so there are no pairs to compare")
    return float(np.mean(values))
