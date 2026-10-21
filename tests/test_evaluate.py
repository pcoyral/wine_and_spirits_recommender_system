import numpy as np
import pandas as pd
import pytest

from sensoryrec.evaluate import evaluate, item_popularity, winery_counts
from sensoryrec.recommenders.attribute import AttributeRecommender
from sensoryrec.recommenders.popularity import PopularityRecommender
from sensoryrec.recommenders.semantic import SemanticRecommender
from sensoryrec.recommenders.tfidf import TfidfRecommender


def wines():
    # Winery A: 3 wines, B: 2, C: 1 (wine 6, a held-out test wine)
    return pd.DataFrame({
        "wine_id": [1, 2, 3, 4, 5, 6],
        "winery": ["A", "A", "A", "B", "B", "C"],
        "variety": ["Pinot Noir", "Pinot Noir", "Riesling", "Riesling", "Merlot", "Merlot"],
        "country": ["US", "US", "Germany", "Germany", "France", "France"],
        "points": [90, 88, 85, 92, 87, 91],
        "price": [15.0, 25.0, 12.0, 30.0, 18.0, 40.0],
        "description": ["red cherry", "red plum", "white lemon",
                        "white peach", "red berry", "red cherry plum"],
    })


HOLDOUT = {5, 6}


def test_winery_counts_ignore_holdout():
    # Without wines 5 and 6: A = 3, B = 1, C = 0
    assert winery_counts(wines(), HOLDOUT).tolist() == [3, 3, 3, 1, 1, 0]


def test_item_popularity_add_one():
    # counted reviews = 4 (wines 1-4), wineries = 3 -> denominator 4 + 3 = 7
    # A: (3 + 1) / 7,  B: (1 + 1) / 7,  C: (0 + 1) / 7  -> sums to 1 over wineries
    p = item_popularity(wines(), HOLDOUT)
    assert p[1] == pytest.approx(4 / 7)
    assert p[4] == pytest.approx(2 / 7)
    assert p[6] == pytest.approx(1 / 7)  # > 0 although C has only a held-out wine
    assert p[1] + p[4] + p[6] == pytest.approx(1.0)


class FakeEncoder:
    WORDS = ["red", "white", "cherry", "plum", "lemon", "peach", "berry"]

    def encode(self, texts, **kwargs):
        return np.array([[t.split().count(w) for w in self.WORDS] for t in texts], dtype=float) + 1e-3


def test_evaluate_end_to_end():
    w = wines()
    colours = pd.DataFrame({"variety": ["Pinot Noir", "Riesling", "Merlot"],
                            "colour": ["red", "white", "red"]})
    models = [PopularityRecommender(), TfidfRecommender(min_df=1),
              AttributeRecommender(colours), SemanticRecommender("fake", encoder=FakeEncoder())]
    for m in models:
        m.fit(w, holdout_ids=HOLDOUT)
    queries = pd.DataFrame({"wine_id": [5, 6], "query": ["red berry", "cherry and plum"],
                            "band": ["1-10", "11-100"]})

    overall, by_band = evaluate(models, w, queries, HOLDOUT, models[-1].vectors(),
                                k=2, tail_max=1, robustness=[0, 3])

    assert overall["model"].tolist() == ["popularity", "tfidf", "attribute", "semantic"]
    assert list(overall.columns) == ["model", "n_queries", "hr@2", "ndcg@2", "coverage", "tail_share",
                                     "tail_share_le0", "tail_share_le3", "gini_wine", "gini_winery",
                                     "novelty", "ild"]
    assert (overall["n_queries"] == 2).all()
    assert len(by_band) == 4 * 2  # 4 models x 2 bands
    # Popularity: winery A (3 counted reviews) first, so the top 2 are wines 1 and 2
    # for both queries; neither target (5, 6) is found and only 2 wines are covered.
    pop = overall.set_index("model").loc["popularity"]
    assert pop["hr@2"] == 0 and pop["coverage"] == pytest.approx(2 / 6)
    # Popularity never recommends a tail wine (winery count <= 1 means B or C)
    assert pop["tail_share"] == 0
    # TF-IDF: "red berry" shares "berry" only with wine 5, so wine 5 ranks first.
    assert by_band.set_index(["model", "band"]).loc[("tfidf", "1-10"), "hr@2"] == 1
