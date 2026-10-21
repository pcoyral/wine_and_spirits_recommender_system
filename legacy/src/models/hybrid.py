"""
Model 4: Hybrid recommender — weighted combination of the other models.

This is the core of the ablation design: setting all weights to zero except
one recovers that individual model (e.g. weights={"sbert": 1.0} reproduces
Model 3 exactly), so Models 0-4 are genuinely special cases of one
parameterized scoring function rather than five separate codebases.
"""

import pandas as pd
from src.models.base import BaseRecommender


class HybridRecommender(BaseRecommender):
    name = "hybrid"

    def __init__(self, catalog: pd.DataFrame, components: dict[str, BaseRecommender], weights: dict[str, float]):
        """
        Parameters
        ----------
        components : dict[str, BaseRecommender]
            e.g. {"popularity": ..., "content_based": ..., "tfidf": ..., "sbert": ...}
        weights : dict[str, float]
            Same keys as components; should sum to 1.0 for interpretability.
        """
        super().__init__(catalog)
        self.components = components
        self.weights = weights

    def fit(self) -> None:
        for model in self.components.values():
            model.fit()

    def score(self, query: dict) -> pd.Series:
        total = pd.Series(0.0, index=self.catalog.index)
        for key, model in self.components.items():
            weight = self.weights.get(key, 0.0)
            if weight == 0:
                continue
            component_scores = model.score(query)
            # TODO: decide on and document a normalization strategy
            # (min-max vs z-score) before combining — raw scores from
            # different models are not on comparable scales.
            total += weight * component_scores
        return total
