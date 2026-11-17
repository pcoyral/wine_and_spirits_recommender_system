"""`make xwines-cf`: collaborative filtering vs content on X-Wines consumer
ratings (scope change of 16 Nov 2026, DECISIONS.md). A separate experiment:
Wine Enthusiast has one review per wine, so CF cannot run on it; X-Wines has
a million users.

Question for the thesis: where users or wines have few ratings (cold start),
does collaborative filtering break down while a content model keeps working?

Data (X-Wines Full, in data/raw/xwines/, as for `make xwines`)
1. A user who rated several vintages of one wine keeps only the latest rating.
2. Users with at least `min_ratings` ratings; a random sample of `n_users`
   of them (fixed seed), so it runs on GitHub's machines.
3. Split by time (leave-last-out): each user's most recent rating is held
   out, the rest is training. Only users whose held-out wine is "liked"
   (rating >= `like_threshold`) are evaluated: the model should put that wine
   in its top k. Wines the user already rated in training are never
   recommended again.

Models (all trained on the training ratings only)
- popularity:  how many users liked the wine
- item_knn:    item-to-item cosine similarity on who liked what; a wine scores
               the sum of its similarities to the wines the user liked
- pure_svd:    matrix factorisation (truncated SVD of the like matrix,
               `n_factors` factors; Cremonesi et al. 2010)
- content:     TF-IDF of each wine's attributes (type, grapes, food pairings,
               body, acidity, country, region); a user is the mean of the
               wines they liked; cosine similarity. Uses no other user.

Metrics at each k: HR, NDCG (one target per user), catalogue coverage,
long-tail share, novelty, Gini. Long tail on X-Wines = wines outside the
"head" that holds 80% of training likes (Pareto split; DECISIONS.md).
Cold-start view: the same metrics per group of users (training ratings) and
per group of target wines (training likes; 0 = a wine nobody liked yet).

Outputs (results/xwines_cf/): summary.csv, by_user_group.csv,
by_item_group.csv, data_summary.csv, fig_hr_by_item_group.png
"""
import ast
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from sensoryrec import metrics
from sensoryrec.config import load_config, repo_path
from sensoryrec.xwines import find_files

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ data
def latest_per_pair(ratings):
    """One rating per (user, wine): the most recent (several vintages)."""
    return (ratings.sort_values(["UserID", "WineID", "Date"])
            .drop_duplicates(["UserID", "WineID"], keep="last").reset_index(drop=True))


def sample_users(ratings, min_ratings, n_users, seed):
    """Ratings of a seeded random sample of users with >= min_ratings ratings."""
    counts = ratings["UserID"].value_counts()
    eligible = counts[counts >= min_ratings].index.sort_values()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
    return ratings[ratings["UserID"].isin(set(chosen))].reset_index(drop=True)


def read_sampled(ratings_path, min_ratings, n_users, seed, chunksize=2_000_000):
    """The ratings of the sampled users, without loading 21 million dates:
    pass 1 reads only user and wine ids to count each user's distinct wines
    and draw the sample; pass 2 reads the full rows in chunks and keeps the
    sampled users only. Same result as sample_users(latest_per_pair(all))."""
    ids = pd.read_csv(ratings_path, usecols=["UserID", "WineID"], dtype={"UserID": "int32", "WineID": "int32"})
    distinct = ids.drop_duplicates()
    chosen = set(sample_users(distinct, min_ratings, n_users, seed)["UserID"])
    del ids, distinct
    parts = [chunk[chunk["UserID"].isin(chosen)]
             for chunk in pd.read_csv(ratings_path, usecols=["UserID", "WineID", "Rating", "Date"],
                                      dtype={"UserID": "int32", "WineID": "int32", "Rating": "float32"},
                                      chunksize=chunksize)]
    return latest_per_pair(pd.concat(parts, ignore_index=True))


def leave_last_out(ratings):
    """(train, test): each user's latest rating is the test rating; ties on
    the date are broken by WineID so the split is always the same."""
    ordered = ratings.sort_values(["UserID", "Date", "WineID"])
    last = ordered.groupby("UserID").tail(1)
    return ordered.drop(last.index).reset_index(drop=True), last.reset_index(drop=True)


def like_matrix(train, like_threshold, n_users, n_items):
    """Binary users x wines matrix: 1 where the user liked the wine.
    train must carry integer columns u and i (row and column positions)."""
    liked = train[train["Rating"] >= like_threshold]
    return sparse.csr_matrix((np.ones(len(liked), dtype=np.float32), (liked["u"], liked["i"])),
                             shape=(n_users, n_items))


def head_items(like_counts, share=0.8):
    """Positions of the most-liked wines that together hold `share` of all
    training likes (the head); every other wine is long tail."""
    order = np.argsort(-like_counts, kind="stable")
    cum = np.cumsum(like_counts[order]) / max(like_counts.sum(), 1)
    n_head = int(np.searchsorted(cum, share) + 1)
    return set(order[:n_head].tolist())


# ------------------------------------------------------------------ models
class Popularity:
    name = "popularity"

    def fit(self, X, item_text=None):
        self.counts = np.asarray(X.sum(axis=0)).ravel()
        return self

    def scores(self, X, users):
        return np.tile(self.counts, (len(users), 1))


class ItemKNN:
    """score(u, j) = sum over wines i liked by u of cos(i, j), where cos is the
    cosine between the columns of the like matrix. Computed as
    (X_u Xn^T) Xn with Xn the column-normalised matrix, so the item-item
    matrix is never built."""
    name = "item_knn"

    def fit(self, X, item_text=None):
        self.Xn = normalize(X, norm="l2", axis=0).tocsr().astype(np.float32)
        return self

    def scores(self, X, users):
        # x_u Xn^T Xn: first (wines liked by u) x Xn^T gives one value per
        # training user, then x Xn spreads it back over the wines.
        through_users = X[users] @ self.Xn.T
        return np.asarray((through_users @ self.Xn).todense())


class PureSVD:
    name = "pure_svd"

    def __init__(self, n_factors=64, seed=42):
        self.n_factors, self.seed = n_factors, seed

    def fit(self, X, item_text=None):
        k = min(self.n_factors, min(X.shape) - 1)
        rng = np.random.default_rng(self.seed)
        v0 = rng.standard_normal(min(X.shape))
        _, _, vt = svds(X.astype(np.float64), k=k, v0=v0)
        self.V = vt.T.astype(np.float32)                 # wines x factors
        return self

    def scores(self, X, users):
        return np.asarray((X[users] @ self.V) @ self.V.T)


class Content:
    """Profile = mean TF-IDF vector of the wines the user liked; score =
    cosine with every wine. Uses no other user's ratings."""
    name = "content"

    def fit(self, X, item_text=None):
        self.V = normalize(TfidfVectorizer(token_pattern=r"[^ ]+").fit_transform(item_text)).astype(np.float32)
        return self

    def scores(self, X, users):
        profiles = normalize(X[users] @ self.V)
        return np.asarray((profiles @ self.V.T).todense())


def wine_text(wines):
    """One "document" per wine from its X-Wines attributes; multi-word values
    become single tokens (e.g. Full-bodied -> body=full-bodied)."""
    def tokens(prefix, value, is_list=False):
        if not isinstance(value, str) or not value:
            return []
        values = ast.literal_eval(value) if is_list else [value]
        return [f"{prefix}={str(v).strip().lower().replace(' ', '_')}" for v in values]
    docs = []
    for r in wines.itertuples():
        t = (tokens("type", r.Type) + tokens("grape", r.Grapes, True) + tokens("food", r.Harmonize, True)
             + tokens("body", r.Body) + tokens("acidity", r.Acidity) + tokens("country", r.Country)
             + tokens("region", r.RegionName))
        docs.append(" ".join(t) or "unknown")
    return docs


# ------------------------------------------------------------------ evaluation
def top_k(scores, exclude, k):
    """Top k positions per row after removing the excluded positions (the
    user's training wines); ties broken by position."""
    s = scores.astype(np.float64, copy=True)
    rows, cols = exclude.nonzero()
    s[rows, cols] = -np.inf
    order = np.argsort(-s, axis=1, kind="stable")
    return order[:, :k]


def group_of(value, edges, labels):
    """Label of the first bucket whose upper edge is >= value."""
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


USER_EDGES, USER_LABELS = [9, 29, np.inf], ["4-9 ratings", "10-29", "30+"]
ITEM_EDGES, ITEM_LABELS = [0, 9, 99, np.inf], ["0 likes (cold)", "1-9", "10-99", "100+"]


def evaluate(models, X, rated, targets, user_train_counts, like_counts, head, k_values, batch=200):
    """Run every model for every evaluated user. targets: user row -> target
    wine position. Returns (summary, per-user rows)."""
    users = np.array(sorted(targets))
    n_items = X.shape[1]
    p = (like_counts + 1) / (like_counts.sum() + n_items)        # add-one popularity share, for novelty
    p_map = dict(enumerate(p))
    rows, summary = [], []
    for model in models:
        recs = []
        for start in range(0, len(users), batch):
            b = users[start:start + batch]
            recs += top_k(model.scores(X, b), rated[b], max(k_values)).tolist()
        log.info("%s: %d users ranked", model.name, len(recs))
        tgt = [targets[u] for u in users]
        for k in k_values:
            rk = [r[:k] for r in recs]
            hits = [t in r for t, r in zip(tgt, rk)]
            summary.append({"model": model.name, "k": k, "n_users": len(users),
                            "hr": metrics.hit_rate_at_k(rk, tgt, k), "ndcg": metrics.ndcg_at_k(rk, tgt, k),
                            "coverage": metrics.catalogue_coverage(rk, n_items),
                            "tail_share": metrics.tail_share(rk, set(range(n_items)) - head),
                            "novelty": metrics.novelty(rk, p_map),
                            "gini": metrics.recommendation_gini(rk, range(n_items))})
            for u, t, r, h in zip(users, tgt, rk, hits):
                rows.append({"model": model.name, "k": k, "user": u, "hit": h,
                             "ndcg": 1 / np.log2(r.index(t) + 2) if h else 0.0,
                             "user_group": group_of(user_train_counts[u], USER_EDGES, USER_LABELS),
                             "item_group": group_of(like_counts[t], ITEM_EDGES, ITEM_LABELS)})
    return pd.DataFrame(summary), pd.DataFrame(rows)


def by_group(per_user, col, labels):
    g = per_user.groupby(["model", "k", col]).agg(n_users=("hit", "size"), hr=("hit", "mean"),
                                                   ndcg=("ndcg", "mean")).reset_index()
    g[col] = pd.Categorical(g[col], labels, ordered=True)
    return g.sort_values(["k", col, "model"]).reset_index(drop=True)


def plot_by_item_group(table, k, path):
    t = table[table["k"] == k]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    models = list(dict.fromkeys(t["model"]))
    width = 0.8 / len(models)
    x = np.arange(len(ITEM_LABELS))
    for i, m in enumerate(models):
        hr = t[t["model"] == m].set_index("item_group")["hr"].reindex(ITEM_LABELS)
        ax.bar(x + i * width - 0.4 + width / 2, hr.fillna(0) * 100, width, label=m)
    ax.set_xticks(x, ["target liked by\n" + lab for lab in ITEM_LABELS])
    ax.set_ylabel(f"HR@{k} (%)")
    ax.set_title("X-Wines: does the model find wines few people have liked?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    c = cfg["xwines_cf"]
    wines_path, ratings_path = find_files(repo_path(cfg["xwines"]["dir"]))
    out = repo_path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    wines = pd.read_csv(wines_path, usecols=["WineID", "Type", "Grapes", "Harmonize", "Body", "Acidity",
                                             "Country", "RegionName"])
    ratings = read_sampled(ratings_path, c["min_ratings"], c["n_users"], cfg["seed"])
    train, test = leave_last_out(ratings)
    test = test[test["Rating"] >= c["like_threshold"]]
    if len(test) > c["n_eval_users"]:
        test = test.sample(n=c["n_eval_users"], random_state=cfg["seed"])

    wine_pos = {w: i for i, w in enumerate(wines["WineID"])}
    user_ids = np.sort(ratings["UserID"].unique())
    user_pos = {u: i for i, u in enumerate(user_ids)}
    for df in (train, test):
        df["u"] = df["UserID"].map(user_pos)
        df["i"] = df["WineID"].map(wine_pos)
    n_users, n_items = len(user_ids), len(wines)
    X = like_matrix(train, c["like_threshold"], n_users, n_items)
    rated = sparse.csr_matrix((np.ones(len(train)), (train["u"], train["i"])), shape=(n_users, n_items))
    like_counts = np.asarray(X.sum(axis=0)).ravel()
    head = head_items(like_counts, c["head_share"])
    user_train_counts = np.asarray(rated.sum(axis=1)).ravel()
    targets = dict(zip(test["u"], test["i"]))

    pd.DataFrame([("users_sampled", n_users), ("ratings_used", len(ratings)), ("train_ratings", len(train)),
                  ("train_likes", int(X.sum())), ("users_evaluated", len(targets)), ("wines", n_items),
                  ("head_wines", len(head)), ("wines_never_liked_in_train", int((like_counts == 0).sum()))],
                 columns=["metric", "value"]).to_csv(out / "data_summary.csv", index=False)

    models = [Popularity(), ItemKNN(), PureSVD(c["n_factors"], cfg["seed"]), Content()]
    text = wine_text(wines)
    for m in models:
        log.info("fitting %s", m.name)
        m.fit(X, text)
    summary, per_user = evaluate(models, X, rated, targets, user_train_counts, like_counts, head, c["k_values"])
    summary.round(4).to_csv(out / "summary.csv", index=False)
    by_group(per_user, "user_group", USER_LABELS).round(4).to_csv(out / "by_user_group.csv", index=False)
    items = by_group(per_user, "item_group", ITEM_LABELS)
    items.round(4).to_csv(out / "by_item_group.csv", index=False)
    plot_by_item_group(items, max(c["k_values"]), out / "fig_hr_by_item_group.png")
    log.info("\n%s", summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
