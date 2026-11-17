import numpy as np
import pandas as pd

from sensoryrec.evaluate import item_popularity, winery_counts
from sensoryrec.subcatalogue import draw_ids, fit_models, subcatalogue_config


def wines(n=20):
    return pd.DataFrame({"wine_id": range(100, 100 + n), "winery": [f"w{i % 4}" for i in range(n)]})


def test_draw_keeps_the_given_wines_and_fills_up_to_size():
    ids = draw_ids(wines(), keep_ids=[105, 117], size=6, seed=42)
    assert len(ids) == 6 and {105, 117} <= set(ids)
    assert ids == sorted(ids)                       # catalogue order
    assert ids == draw_ids(wines(), [105, 117], 6, 42)   # same seed, same wines


def test_draw_refuses_a_size_smaller_than_the_kept_wines():
    import pytest
    with pytest.raises(ValueError):
        draw_ids(wines(), keep_ids=[100, 101, 102], size=2, seed=42)


def test_config_switches_gemini_on_and_moves_the_cache_without_touching_the_original():
    cfg = {"semantic": {"cache_dir": "full"}, "subcatalogue": {"embeddings_dir": "sub"},
           "extra_embedding_models": [{"name": "semantic_gemini", "enabled": False}]}
    sub = subcatalogue_config(cfg)
    assert sub["semantic"]["cache_dir"] == "sub" and sub["extra_embedding_models"][0]["enabled"] is True
    assert cfg["semantic"]["cache_dir"] == "full" and cfg["extra_embedding_models"][0]["enabled"] is False


class FakeModel:
    def __init__(self, name, log):
        self.name, self.log, self.query_prefix = name, log, ""

    def fit(self, wines, holdout_ids=()):
        self.log.append(("fit", self.name))

    def encode_queries(self, texts):
        self.log.append(("queries", self.name, len(texts)))


def test_gemini_is_fitted_first_and_embeds_the_queries_at_once():
    log = []
    models = [FakeModel("tfidf", log), FakeModel("semantic", log), FakeModel("semantic_gemini", log)]
    fit_models(models, wines(), [], queries=["a", "b", "c"])
    assert log[:2] == [("fit", "semantic_gemini"), ("queries", "semantic_gemini", 3)]
    assert [m.name for m in models] == ["tfidf", "semantic", "semantic_gemini"]   # table order unchanged


def test_tail_counts_can_come_from_the_full_catalogue():
    full = wines(20)                                 # each winery has 5 reviews
    sub = full.iloc[[0, 1]]                          # one review each in the sub-catalogue
    assert winery_counts(sub, holdout_ids=[]).tolist() == [1, 1]
    assert winery_counts(sub, holdout_ids=[], reference=full).tolist() == [5, 5]
    p = item_popularity(sub, holdout_ids=[], reference=full)
    assert np.isclose(p[100], (5 + 1) / (20 + 4))
