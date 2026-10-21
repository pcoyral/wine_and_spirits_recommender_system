"""Shared interface for every recommender.

Each model implements fit() and score(); recommend() is the same for all:
score every wine, apply the filters, take the top k. That keeps the models
comparable: they differ only in how they score.
"""
from abc import ABC, abstractmethod

import numpy as np


class Recommender(ABC):
    name = "base"

    def fit(self, catalogue, holdout_ids=()):
        """Store the catalogue. Subclasses learn whatever they need from it,
        ignoring the wines in `holdout_ids` (the test set, CLAUDE.md rule 4).
        The holdout wines stay in the catalogue: they can still be recommended."""
        self.catalogue = catalogue.reset_index(drop=True)
        self.holdout_ids = set(holdout_ids)
        return self

    @abstractmethod
    def score(self, query):
        """One score per catalogue row (higher = better match for `query`)."""

    def recommend(self, query, k=5, filters=None):
        """The k best wines for `query` that pass `filters`, best first.
        Ties are broken by wine_id, so the result is always the same."""
        return rank(self.catalogue, self.score(query), k, filters)


def rank(catalogue, scores, k=5, filters=None):
    """The k best-scoring wines that pass `filters`, best first, ties broken
    by wine_id. Shared by every model (and by refinement)."""
    result = catalogue.assign(score=np.asarray(scores, dtype=float))
    result = result[filter_mask(result, filters)]
    result = result.sort_values(["score", "wine_id"], ascending=[False, True]).head(k)
    return result.assign(rank=np.arange(1, len(result) + 1))


FILTER_KEYS = {"max_price", "min_price", "country", "variety"}


def filter_mask(df, filters):
    """True for wines that pass every filter. Supported keys: max_price,
    min_price, country, variety. A wine without a price never passes a price
    filter: its price is unknown, not cheap (DECISIONS.md). An unknown key is
    an error, so a misspelled filter is never silently ignored."""
    mask = np.ones(len(df), dtype=bool)
    if not filters:
        return mask
    unknown = set(filters) - FILTER_KEYS
    if unknown:
        raise ValueError(f"unknown filter(s) {sorted(unknown)}; supported: {sorted(FILTER_KEYS)}")
    if "max_price" in filters:
        mask &= (df["price"] <= filters["max_price"]).fillna(False).to_numpy()
    if "min_price" in filters:
        mask &= (df["price"] >= filters["min_price"]).fillna(False).to_numpy()
    for col in ["country", "variety"]:
        if col in filters:
            mask &= (df[col] == filters[col]).to_numpy()
    return mask
