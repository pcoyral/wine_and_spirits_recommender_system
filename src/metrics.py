"""
Evaluation metrics for SensoryRec.

Per the development roadmap, this should be built in parallel with the
hybrid recommender (Milestones 2-4) to resolve the circular dependency
between having a model to evaluate and having an evaluation framework to
validate models against.

Evaluation protocol: self-retrieval using paraphrased natural-language
queries against held-out wines (no real user interaction data exists),
validated by sommelier panel review.
"""

import numpy as np
import pandas as pd


def ndcg_at_k(recommended_ids: list, relevant_ids: set, k: int = 5) -> float:
    """
    NDCG@k for a single query.

    Parameters
    ----------
    recommended_ids : list
        Ranked list of item ids returned by a model, best-first.
    relevant_ids : set
        Ground-truth relevant item ids (e.g. the held-out wine + accepted
        paraphrase matches).
    """
    recommended_ids = recommended_ids[:k]
    gains = [1.0 if item in relevant_ids else 0.0 for item in recommended_ids]
    discounts = [1.0 / np.log2(i + 2) for i in range(len(gains))]
    dcg = sum(g * d for g, d in zip(gains, discounts))

    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g * d for g, d in zip(ideal_gains, discounts))
    return dcg / idcg if idcg > 0 else 0.0


def serendipity(recommended_ids: list, expected_ids: set, catalog: pd.DataFrame) -> float:
    """
    Placeholder serendipity score: fraction of recommendations that are
    relevant but NOT in the "obvious"/expected set (e.g. not same
    winery/variety as a naive popularity baseline would suggest).

    TODO: finalize the definition of "expected" against the literature
    review's beyond-accuracy metrics section before relying on this.
    """
    raise NotImplementedError


def evaluate_model(model, eval_queries: list[dict], k: int = 5) -> pd.DataFrame:
    """
    Run a model against a set of evaluation queries and return per-query
    and aggregate metrics.

    eval_queries : list of dicts, each with at least:
        - "query": dict passed to model.score()
        - "relevant_ids": set of ground-truth relevant item ids
    """
    rows = []
    for eq in eval_queries:
        recs = model.recommend(eq["query"], k=k)
        recommended_ids = list(recs.index)
        score = ndcg_at_k(recommended_ids, eq["relevant_ids"], k=k)
        rows.append({"model": model.name, "ndcg_at_k": score})
    return pd.DataFrame(rows)
