"""Popularity baseline: ignores the query and recommends wines from the
most-reviewed wineries.

Each wine has exactly one review, so "popular" is measured at winery level:
the number of reviews of the wine's winery, the same unit as the long-tail
definition (DECISIONS.md). Points break ties between wines of the same winery.
"""
from sensoryrec.recommenders.base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def fit(self, catalogue, holdout_ids=()):
        super().fit(catalogue, holdout_ids)
        # Count reviews per winery without the test wines (no leakage).
        seen = self.catalogue[~self.catalogue["wine_id"].isin(self.holdout_ids)]
        counts = seen["winery"].value_counts()
        self.winery_reviews = self.catalogue["winery"].map(counts).fillna(0).to_numpy()
        # Whole part = winery review count, decimals = points / 1000.
        # Points are at most 100, so they can only break ties, never
        # overtake a winery with one more review.
        self._scores = self.winery_reviews + self.catalogue["points"].to_numpy() / 1000
        return self

    def score(self, query):
        # Same scores for every query: that is what makes it a baseline.
        return self._scores
