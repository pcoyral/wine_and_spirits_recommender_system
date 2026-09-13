"""
Model 0: Popularity baseline (non-personalized).

Per the development roadmap, this should be built first since metrics.py
(Milestone 6) needs a working end-to-end model to validate the evaluation
protocol against, in parallel with Milestones 2-4.
"""

import pandas as pd
from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def __init__(self, catalog: pd.DataFrame, popularity_col: str = "points"):
        super().__init__(catalog)
        self.popularity_col = popularity_col
        self._scores: pd.Series | None = None

    def fit(self) -> None:
        # TODO: replace with a proper popularity signal once defined
        # (e.g. review count, points, or a composite). Using `points`
        # as a placeholder proxy.
        self._scores = self.catalog[self.popularity_col].astype(float)

    def score(self, query: dict) -> pd.Series:
        if self._scores is None:
            raise RuntimeError("Call fit() before score().")
        # Popularity is query-independent by definition.
        return self._scores
