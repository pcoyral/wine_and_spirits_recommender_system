import numpy as np
import pandas as pd
import pytest

from sensoryrec.recommenders.base import filter_mask
from sensoryrec.recommenders.popularity import PopularityRecommender
from sensoryrec.recommenders import build_models, ild_vectors
from sensoryrec.recommenders.semantic import SemanticRecommender, extra_from_config


def catalogue():
    # Winery A has 3 wines, B has 2, C has 1.
    return pd.DataFrame({
        "wine_id": [10, 11, 12, 20, 21, 30],
        "winery": ["A", "A", "A", "B", "B", "C"],
        "points": [90, 85, 88, 92, 87, 95],
        "price": [10.0, 30.0, np.nan, 15.0, 25.0, 50.0],
        "country": ["US", "US", "US", "France", "France", "Italy"],
        "variety": ["Merlot"] * 6,
        "description": ["red red", "white", "bubbly white", "red", "white red", "bubbly"],
    })


def test_popularity_ranks_by_winery_then_points():
    model = PopularityRecommender().fit(catalogue())
    top = model.recommend("anything", k=4)
    # A (3 reviews) before B (2); inside A by points: 10 (90), 12 (88), 11 (85)
    assert list(top["wine_id"]) == [10, 12, 11, 20]
    assert list(top["rank"]) == [1, 2, 3, 4]


def test_popularity_ignores_query():
    model = PopularityRecommender().fit(catalogue())
    a = model.recommend("light white", k=3)["wine_id"].tolist()
    b = model.recommend("big red", k=3)["wine_id"].tolist()
    assert a == b


def test_popularity_does_not_count_holdout_wines():
    # Hold out two of A's wines: A now has 1 counted review, B has 2.
    model = PopularityRecommender().fit(catalogue(), holdout_ids=[11, 12])
    top = model.recommend("anything", k=2)
    assert list(top["wine_id"]) == [20, 21]  # B first now
    # The held-out wines are still in the catalogue and can be recommended.
    assert {11, 12} <= set(model.recommend("anything", k=6)["wine_id"])


def test_filters():
    df = catalogue()
    # Max 20: wines 10 (10.0) and 20 (15.0); wine 12 has no price, so it is out.
    assert df[filter_mask(df, {"max_price": 20})]["wine_id"].tolist() == [10, 20]
    assert df[filter_mask(df, {"country": "France"})]["wine_id"].tolist() == [20, 21]
    assert filter_mask(df, None).all()


def test_unknown_filter_is_an_error():
    # A misspelled key must not silently return unfiltered wines.
    with pytest.raises(ValueError, match="price_max"):
        filter_mask(catalogue(), {"price_max": 20})


def test_recommend_applies_filters():
    model = PopularityRecommender().fit(catalogue())
    top = model.recommend("anything", k=5, filters={"max_price": 20})
    assert list(top["wine_id"]) == [10, 20]  # fewer than k wines pass


class FakeEncoder:
    """Counts three words, so the vectors (and cosines) can be worked out by hand."""
    WORDS = ["red", "white", "bubbly"]

    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.array([[t.split().count(w) for w in self.WORDS] for t in texts], dtype=float)


def test_semantic_cosine_ranking():
    model = SemanticRecommender("fake", encoder=FakeEncoder()).fit(catalogue())
    # Query "white bubbly" = (0, 1, 1)/sqrt(2). Cosines with the notes:
    #   12 "bubbly white" (0,1,1)        -> 1
    #   11 "white" (0,1,0)               -> 1/sqrt(2) = 0.707
    #   21 "white red" (1,1,0)/sqrt(2)   -> 1/2
    #   30 "bubbly" (0,0,1)              -> 0.707, tie with 11, broken by wine_id
    top = model.recommend("white bubbly", k=4)
    assert list(top["wine_id"]) == [12, 11, 30, 21]
    assert top["score"].tolist() == pytest.approx([1, 1 / np.sqrt(2), 1 / np.sqrt(2), 0.5])


def test_semantic_scores_are_cosines():
    model = SemanticRecommender("fake", encoder=FakeEncoder()).fit(catalogue())
    # "red red" and "red" both point along (1, 0, 0): cosine 1 with query "red"
    scores = dict(zip(model.catalogue["wine_id"], model.score("red")))
    assert scores[10] == pytest.approx(1.0) and scores[20] == pytest.approx(1.0)
    assert scores[11] == pytest.approx(0.0)


def test_semantic_embedding_cache(tmp_path):
    enc = FakeEncoder()
    SemanticRecommender("fake", cache_dir=tmp_path, encoder=enc).fit(catalogue())
    assert enc.calls == 1
    # Same catalogue: loaded from the cache, the encoder is not called again.
    SemanticRecommender("fake", cache_dir=tmp_path, encoder=enc).fit(catalogue())
    assert enc.calls == 1
    # A note changes: the fingerprint no longer matches, so it re-encodes.
    changed = catalogue()
    changed.loc[0, "description"] = "white"
    SemanticRecommender("fake", cache_dir=tmp_path, encoder=enc).fit(changed)
    assert enc.calls == 2


def test_semantic_cache_rebuilt_when_model_version_changes(tmp_path):
    enc = FakeEncoder()
    SemanticRecommender("fake", revision="v1", cache_dir=tmp_path, encoder=enc).fit(catalogue())
    SemanticRecommender("fake", revision="v1", cache_dir=tmp_path, encoder=enc).fit(catalogue())
    assert enc.calls == 1
    # Same notes, new model version: the old embeddings must not be reused.
    SemanticRecommender("fake", revision="v2", cache_dir=tmp_path, encoder=enc).fit(catalogue())
    assert enc.calls == 2


class PrefixEncoder(FakeEncoder):
    """Remembers every text it is asked to encode."""
    def __init__(self):
        super().__init__()
        self.texts = []

    def encode(self, texts, **kwargs):
        self.texts += list(texts)
        return super().encode(texts)


def test_query_prefix_goes_on_the_query_only():
    enc = PrefixEncoder()
    model = SemanticRecommender("fake", encoder=enc, name="semantic_x", query_prefix="search: ").fit(catalogue())
    model.recommend("white bubbly", k=2)
    assert model.name == "semantic_x"
    assert enc.texts[:6] == catalogue()["description"].tolist()   # the notes, unchanged
    assert enc.texts[6:] == ["search: white bubbly"]              # the query, with the instruction


CFG = {"tfidf": {"min_df": 2, "stop_words": "english", "keep_words": ["not"], "sublinear_tf": True,
                 "ngram_range": [1, 1]},
       "attribute": {"colour_table": "data/variety_colour.csv"},
       "semantic": {"model": "sentence-transformers/all-MiniLM-L6-v2", "revision": None,
                    "batch_size": 64, "cache_dir": "data/processed/embeddings"}}
EXTRA = {"name": "semantic_bge", "model": "BAAI/bge-small-en-v1.5", "revision": "abc", "batch_size": 32,
         "query_prefix": "search: "}


def test_extra_embedding_models_from_config():
    assert extra_from_config(CFG) == []                           # the section is optional
    assert extra_from_config({**CFG, "extra_embedding_models": None}) == []
    [bge] = extra_from_config({**CFG, "extra_embedding_models": [EXTRA]})
    assert (bge.name, bge.model_name, bge.revision, bge.batch_size) == \
        ("semantic_bge", "BAAI/bge-small-en-v1.5", "abc", 32)
    assert bge.query_prefix == "search: "
    # its own cache file, next to Sentence-BERT's
    assert bge.cache_path().name == "bge-small-en-v1.5.npz"


class OtherEncoder(FakeEncoder):
    """Another model: different vectors for the same texts."""
    def encode(self, texts, **kwargs):
        return super().encode(texts) + 1.0


def test_build_models_order_and_ild_yardstick():
    models = build_models({**CFG, "extra_embedding_models": [EXTRA]})
    assert [m.name for m in models] == ["popularity", "tfidf", "attribute", "semantic", "semantic_bge"]
    # Diversity is always measured with Sentence-BERT ("semantic"), also when
    # another embedding model comes after it in the list.
    sbert, bge = models[3], models[4]
    sbert._encoder, bge._encoder = FakeEncoder(), OtherEncoder()
    sbert.cache_dir = bge.cache_dir = None
    sbert.fit(catalogue())
    bge.fit(catalogue())
    assert np.array_equal(ild_vectors(models)[11], sbert.vectors()[11])
    assert not np.array_equal(ild_vectors(models)[11], bge.vectors()[11])
