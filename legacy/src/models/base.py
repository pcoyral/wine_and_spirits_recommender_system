"""
Base interface for all SensoryRec recommender models.

Per the project's technical methodology, Models 1-4 (Content-Based, TF-IDF,
Sentence-BERT, Hybrid) are structured as special cases of the same weighted
scoring function. Each model subclasses BaseRecommender and implements
score(), which returns a similarity/relevance score for each candidate item
given a query (structured filters and/or natural-language text).

This lets the full ablation table (Model 0-4) be produced from one
parameterized implementation, and keeps evaluation (src/metrics.py) uniform
across all five architectures.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseRecommender(ABC):
    """Common interface for all recommender models in the ablation ladder."""

    name: str = "base"

    def __init__(self, catalog: pd.DataFrame):
        """
        Parameters
        ----------
        catalog : pd.DataFrame
            The wine catalog (e.g. loaded via src/data_loader.py), indexed
            consistently across all models so recommendations can be compared.
        """
        self.catalog = catalog

    @abstractmethod
    def fit(self) -> None:
        """Precompute anything needed at inference time (indices, embeddings, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def score(self, query: dict) -> pd.Series:
        """
        Score every item in the catalog against a query.

        Parameters
        ----------
        query : dict
            Expected keys depend on the model, but typically include:
            - "text": str, natural-language preference description (optional)
            - "filters": dict, structured attribute filters (optional)

        Returns
        -------
        pd.Series
            Index-aligned with self.catalog, higher score = more relevant.
        """
        raise NotImplementedError

    def recommend(self, query: dict, k: int = 5) -> pd.DataFrame:
        """Return the top-k catalog rows for a given query."""
        scores = self.score(query)
        top_idx = scores.sort_values(ascending=False).head(k).index
        return self.catalog.loc[top_idx]
