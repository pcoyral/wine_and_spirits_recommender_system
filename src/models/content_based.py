"""
Model 1: Content-Based recommender using structured attributes
(variety, region, price bracket, etc.) rather than free text.
"""

import pandas as pd
from src.models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    name = "content_based"

    def __init__(self, catalog: pd.DataFrame, attribute_cols: list[str] | None = None):
        super().__init__(catalog)
        self.attribute_cols = attribute_cols or ["variety", "province", "region_1"]

    def fit(self) -> None:
        # TODO: build a structured similarity index over self.attribute_cols
        # (e.g. one-hot encode categoricals + normalize price/points, then
        # cosine similarity or a nearest-neighbor index).
        raise NotImplementedError

    def score(self, query: dict) -> pd.Series:
        # TODO: given query["filters"], score catalog rows by attribute match.
        raise NotImplementedError
