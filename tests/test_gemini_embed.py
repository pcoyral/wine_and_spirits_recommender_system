import numpy as np
import pandas as pd
import pytest

from sensoryrec.gemini_embed import GeminiEncoder
from sensoryrec.recommenders.semantic import SemanticRecommender, extra_from_config


class FakeAPI:
    """Embeds a text as (len, count of 'red', count of 'white'); can fail after n calls."""
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after

    def vec(self, text):
        return [float(len(text)), float(text.count("red")), float(text.count("white"))]

    def __call__(self, url, body):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise SystemExit("quota")
        self.calls.append((url, body))
        if url.endswith(":batchEmbedContents"):
            assert all(r["taskType"] == "RETRIEVAL_DOCUMENT" and r["outputDimensionality"] == 3 for r in body["requests"])
            return {"embeddings": [{"values": self.vec(r["content"]["parts"][0]["text"])} for r in body["requests"]]}
        assert body["taskType"] == "RETRIEVAL_QUERY"
        return {"embedding": {"values": self.vec(body["content"]["parts"][0]["text"])}}


def encoder(tmp_path, api):
    return GeminiEncoder("gemini-embedding-001", 3, tmp_path / "batches", tmp_path / "cache", batch_size=2, post=api)


def test_documents_in_batches_and_resume_after_a_stop(tmp_path):
    texts = ["red", "white", "red red", "white wine", "red"]
    first = FakeAPI(fail_after=2)              # quota ends after 2 of the 3 batches
    with pytest.raises(SystemExit):
        encoder(tmp_path, first).encode(texts)
    second = FakeAPI()
    out = encoder(tmp_path, second).encode(texts)
    assert len(second.calls) == 1              # only the missing batch is requested again
    assert out.shape == (5, 3) and out[2].tolist() == [7.0, 2.0, 0.0]
    assert all(url.endswith(":batchEmbedContents") for url, _ in first.calls + second.calls)


def test_queries_are_cached(tmp_path):
    api = FakeAPI()
    e = encoder(tmp_path, api)
    a = e.encode_queries(["a red please", "a white please"])
    b = encoder(tmp_path, api).encode_queries(["a red please"])
    assert len(api.calls) == 2 and b[0].tolist() == a[0].tolist()
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2


def test_recommender_uses_the_query_mode(tmp_path):
    api = FakeAPI()
    wines = pd.DataFrame({"wine_id": [1, 2], "description": ["white white", "red red"]})
    model = SemanticRecommender("gemini-embedding-001", encoder=encoder(tmp_path, api), name="semantic_gemini")
    top = model.fit(wines).recommend("red", k=1)
    assert top["wine_id"].tolist() == [2]
    assert api.calls[-1][0].endswith(":embedContent")        # the request went through query mode


def test_gemini_is_off_until_enabled():
    cfg = {"semantic": {"cache_dir": "data/processed/embeddings"}, "llm": {"cache_dir": "data/llm_cache"},
           "extra_embedding_models": [{"name": "semantic_gemini", "provider": "gemini", "model": "gemini-embedding-001",
                                       "revision": "gemini-embedding-001", "dims": 768, "batch_size": 100,
                                       "enabled": False}]}
    assert extra_from_config(cfg) == []
    cfg["extra_embedding_models"][0]["enabled"] = True
    [m] = extra_from_config(cfg)
    assert m.name == "semantic_gemini" and m._encoder.dims == 768


def test_queries_wait_their_own_pause(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr("sensoryrec.gemini_embed.time.sleep", waits.append)
    e = GeminiEncoder("gemini-embedding-001", 3, tmp_path / "batches", tmp_path / "cache", batch_size=2,
                      pause_seconds=60, query_pause_seconds=1, post=FakeAPI())
    e.encode(["red", "white"])
    e.encode_queries(["a red please"])
    assert waits == [60, 1]          # one batch of notes, then one request
