"""
Model 3: Sentence-BERT recommender — dense semantic embeddings for the
user's natural-language query and each wine's `description`, compared via
cosine similarity. This is the core of the "vocabulary-agnostic
personalization" mechanism (e.g. "fruity but not that fruity" should match
semantically even without exact keyword overlap, unlike Model 2/TF-IDF).
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.models.base import BaseRecommender


class SBertRecommender(BaseRecommender):
    name = "sbert"

    def __init__(
        self,
        catalog: pd.DataFrame,
        text_col: str = "description",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__(catalog)
        self.text_col = text_col
        self.model = SentenceTransformer(model_name)
        self._embeddings: np.ndarray | None = None

    def fit(self) -> None:
        texts = self.catalog[self.text_col].fillna("").tolist()
        # TODO: cache embeddings to disk (models_cache/, gitignored) — encoding
        # ~130k descriptions repeatedly during development will be slow.
        self._embeddings = self.model.encode(texts, show_progress_bar=True)

    def score(self, query: dict) -> pd.Series:
        if self._embeddings is None:
            raise RuntimeError("Call fit() before score().")
        text = query.get("text", "")
        query_emb = self.model.encode([text])
        sims = (self._embeddings @ query_emb.T).flatten()
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb)
        sims = sims / np.clip(norms, 1e-8, None)
        return pd.Series(sims, index=self.catalog.index)
