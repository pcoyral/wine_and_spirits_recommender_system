import numpy as np
import pandas as pd
import pytest

from sensoryrec.evaluate import (coverage_curve, evaluate, evaluate_simulated, item_popularity,
                                 load_queries, load_simulated, winery_counts)
from sensoryrec.recommenders import ild_vectors
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


def fitted_models(w):
    colours = pd.DataFrame({"variety": ["Pinot Noir", "Riesling", "Merlot"],
                            "colour": ["red", "white", "red"]})
    models = [PopularityRecommender(), TfidfRecommender(min_df=1),
              AttributeRecommender(colours), SemanticRecommender("fake", encoder=FakeEncoder())]
    for m in models:
        m.fit(w, holdout_ids=HOLDOUT)
    return models


def test_evaluate_end_to_end():
    w = wines()
    models = fitted_models(w)
    queries = pd.DataFrame({"wine_id": [5, 6], "query": ["red berry", "cherry and plum"],
                            "band": ["1-10", "11-100"]})

    overall, by_band = evaluate(models, w, queries, HOLDOUT, ild_vectors(models),
                                k=2, tail_max=1, robustness=[0, 3])

    assert overall["model"].tolist() == ["popularity", "tfidf", "attribute", "semantic"]
    assert list(overall.columns) == ["model", "n_queries", "hr@2", "ndcg@2", "coverage", "winery_coverage",
                                     "variety_coverage", "tail_share", "tail_share_le0", "tail_share_le3",
                                     "gini_wine", "gini_winery", "novelty", "ild"]
    assert (overall["n_queries"] == 2).all()
    assert len(by_band) == 4 * 2  # 4 models x 2 bands
    # Popularity: winery A (3 counted reviews) first, so the top 2 are wines 1 and 2
    # for both queries; neither target (5, 6) is found and only 2 wines are covered.
    pop = overall.set_index("model").loc["popularity"]
    assert pop["hr@2"] == 0 and pop["coverage"] == pytest.approx(2 / 6)
    # Wines 1 and 2 are both winery A (1 of 3 wineries) and both Pinot Noir (1 of 3 varieties)
    assert pop["winery_coverage"] == pytest.approx(1 / 3) and pop["variety_coverage"] == pytest.approx(1 / 3)
    # Popularity never recommends a tail wine (winery count <= 1 means B or C)
    assert pop["tail_share"] == 0
    # TF-IDF: "red berry" shares "berry" only with wine 5, so wine 5 ranks first.
    assert by_band.set_index(["model", "band"]).loc[("tfidf", "1-10"), "hr@2"] == 1


def test_load_queries_leaves_out_flagged(tmp_path):
    pd.DataFrame({"wine_id": [5, 6], "query": ["ok", "a merlot"], "flagged": [False, True]}) \
        .to_csv(tmp_path / "test_queries.csv", index=False)
    split_ids = pd.DataFrame({"wine_id": [5, 6], "band": ["1-10", "11-100"]})
    cfg = {"eval": {"queries_dir": tmp_path}}
    queries = load_queries(cfg, "test", split_ids)
    assert queries["wine_id"].tolist() == [5] and queries["band"].tolist() == ["1-10"]


def test_load_queries_without_file_stops(tmp_path):
    with pytest.raises(SystemExit, match="No casual queries"):
        load_queries({"eval": {"queries_dir": tmp_path}}, "test", pd.DataFrame({"wine_id": []}))


def test_coverage_curve():
    # Distinct wines after 2, 4 and (the last) 5 queries: {1,2}, {1,2,3}, {1,2,3,9}
    recs = [[1, 2], [1, 2], [2, 3], [3, 1], [9, 1]]
    assert coverage_curve(recs, n_wines=10, step=2) == [
        {"n_queries": 2, "distinct_wines": 2, "coverage": 0.2},
        {"n_queries": 4, "distinct_wines": 3, "coverage": 0.3},
        {"n_queries": 5, "distinct_wines": 4, "coverage": 0.4}]


def test_evaluate_simulated_has_no_accuracy_columns():
    w = wines()
    models = fitted_models(w)
    queries = pd.DataFrame({"query": ["red cherry", "white lemon", "something for a picnic"]})

    table, curve = evaluate_simulated(models, w, queries, HOLDOUT, ild_vectors(models),
                                      k=2, tail_max=1, robustness=[0, 3], step=2)

    assert list(table.columns) == ["model", "n_queries", "slots", "distinct_wines",
                                   "coverage", "winery_coverage", "variety_coverage",
                                   "tail_share", "tail_share_le0", "tail_share_le3", "gini_wine",
                                   "gini_winery", "novelty", "ild"]
    assert (table["n_queries"] == 3).all()
    # Popularity answers every query with wines 1 and 2: the curve stays flat at 2 wines
    pop = curve[curve["model"] == "popularity"]
    assert pop["n_queries"].tolist() == [2, 3] and pop["distinct_wines"].tolist() == [2, 2]
    pop_row = table.set_index("model").loc["popularity"]
    assert pop_row["slots"] == 6 and pop_row["distinct_wines"] == 2   # 3 queries x k = 2
    assert pop_row["coverage"] == pytest.approx(2 / 6)
    # TF-IDF follows the words, so it reaches more wines than Popularity
    tfidf = curve[curve["model"] == "tfidf"]
    assert tfidf["distinct_wines"].iloc[-1] > 2


def test_load_simulated_leaves_out_flagged(tmp_path):
    cfg = {"simulation": {"out": tmp_path / "simulated_queries.csv"}}
    with pytest.raises(SystemExit, match="make simulate"):
        load_simulated(cfg)
    pd.DataFrame({"query_id": [1, 2], "query": ["ok", "firm tannins"], "flagged": [False, True]}) \
        .to_csv(cfg["simulation"]["out"], index=False)
    assert load_simulated(cfg)["query"].tolist() == ["ok"]
