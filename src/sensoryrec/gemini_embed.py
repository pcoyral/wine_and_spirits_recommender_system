"""Gemini embeddings as a drop-in encoder for SemanticRecommender (step 4 of
the plan of 16 Nov 2026; DECISIONS.md): `gemini-embedding-001`, 768
dimensions, through the HTTP API (key in GEMINI_API_KEY, never in the repo).

Two kinds of text, as the API recommends:
- tasting notes -> task type RETRIEVAL_DOCUMENT, sent 100 per request
  (batchEmbedContents). Each finished batch is saved at once in
  <cache_dir>/gemini_batches/, so a run stopped by the free quota continues
  where it stopped.
- requests -> task type RETRIEVAL_QUERY, one per call, each answer cached as
  JSON in data/llm_cache/ (committed, like every other LLM output).
Vectors shorter than 3,072 dimensions must be normalised; unit_vectors() in
the recommender does that.
"""
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from sensoryrec.llm import post_json

log = logging.getLogger(__name__)
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}"


class GeminiEncoder:
    def __init__(self, model, dims, batch_dir, query_cache_dir, batch_size=100, pause_seconds=0, post=None):
        """post: function (url, body) -> parsed JSON. Left as None, the real
        API is called; tests pass a fake."""
        self.model, self.dims, self.batch_size, self.pause = model, dims, batch_size, pause_seconds
        self.batch_dir, self.query_cache_dir = Path(batch_dir), Path(query_cache_dir)
        self.post = post or post_json
        self.n_calls = 0

    def _request(self, text, task):
        return {"model": f"models/{self.model}", "content": {"parts": [{"text": text}]},
                "taskType": task, "outputDimensionality": self.dims}

    def _key(self, texts, task):
        data = json.dumps({"model": self.model, "dims": self.dims, "task": task, "texts": list(texts)})
        return hashlib.sha256(data.encode()).hexdigest()

    def _call(self, method, body):
        out = self.post(API.format(model=self.model, method=method), body)
        self.n_calls += 1
        time.sleep(self.pause)
        return out

    def encode(self, texts, batch_size=None, show_progress_bar=False):
        """Tasting notes (documents), batch by batch, resumable."""
        texts = list(texts)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        size = batch_size or self.batch_size
        parts = []
        for start in range(0, len(texts), size):
            batch = texts[start:start + size]
            path = self.batch_dir / f"{self._key(batch, 'RETRIEVAL_DOCUMENT')}.npy"
            if not path.exists():
                data = self._call("batchEmbedContents",
                                  {"requests": [self._request(t, "RETRIEVAL_DOCUMENT") for t in batch]})
                vectors = np.array([e["values"] for e in data["embeddings"]], dtype=np.float32)
                if vectors.shape != (len(batch), self.dims):
                    raise ValueError(f"unexpected embedding shape {vectors.shape}")
                np.save(path, vectors)
            parts.append(np.load(path))
            if (start // size) % 50 == 0:
                log.info("gemini: %d / %d notes embedded (%d new requests)", start + len(batch), len(texts), self.n_calls)
        return np.concatenate(parts) if parts else np.zeros((0, self.dims), dtype=np.float32)

    def encode_queries(self, texts):
        """Requests, one call each, cached as JSON (committed)."""
        self.query_cache_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for text in texts:
            path = self.query_cache_dir / f"{self._key([text], 'RETRIEVAL_QUERY')}.json"
            if path.exists():
                out.append(json.loads(path.read_text())["embedding"])
                continue
            data = self._call("embedContent", self._request(text, "RETRIEVAL_QUERY"))
            vector = data["embedding"]["values"]
            path.write_text(json.dumps({"model": self.model, "dims": self.dims, "task": "RETRIEVAL_QUERY",
                                        "text": text, "embedding": vector}))
            out.append(vector)
        return np.array(out, dtype=np.float32)
