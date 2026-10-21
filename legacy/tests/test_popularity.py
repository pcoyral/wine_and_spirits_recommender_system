import pandas as pd
from src.models.popularity import PopularityRecommender


def test_popularity_recommends_highest_scoring_items():
    catalog = pd.DataFrame({
        "title": ["Wine A", "Wine B", "Wine C"],
        "points": [85, 95, 90],
    })
    model = PopularityRecommender(catalog, popularity_col="points")
    model.fit()
    top = model.recommend(query={}, k=1)
    assert top.iloc[0]["title"] == "Wine B"
