"""
Model 2: TF-IDF recommender — sparse lexical similarity between the user's
natural-language query and each wine's `description` field.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.models.base import BaseRecommender


class TfidfRecommender(BaseRecommender):
    name = "tfidf"

    def __init__(self, catalog: pd.DataFrame, text_col: str = "description"):
        super().__init__(catalog)
        self.text_col = text_col
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None

    def fit(self) -> None:
        texts = self.catalog[self.text_col].fillna("")
        self._matrix = self.vectorizer.fit_transform(texts)

    def score(self, query: dict) -> pd.Series:
        if self._matrix is None:
            raise RuntimeError("Call fit() before score().")
        text = query.get("text", "")
        query_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self._matrix).flatten()
        return pd.Series(sims, index=self.catalog.index)
