"""Semantic recommender: Sentence-BERT embeddings + cosine similarity.

The query and every tasting note are turned into vectors by the same
pretrained model (all-MiniLM-L6-v2); wines whose note points in the most
similar direction to the query rank highest. Vectors are scaled to length 1,
so cosine similarity is simply a dot product.

The model is pretrained and learns nothing from our data, so there is
nothing to hold out: test wines are embedded too, as retrieval targets.

Embeddings for the ~120k notes take minutes to compute, so they are cached in
data/processed/embeddings/. The cache stores a fingerprint of the model name,
the wine_ids and the notes; if any of these change, it is rebuilt.

Run `make embed` to build the cache.
"""
import hashlib
import logging

import numpy as np
import pandas as pd

from sensoryrec.config import load_config, repo_path
from sensoryrec.recommenders.base import Recommender

log = logging.getLogger(__name__)


def unit_vectors(m):
    """Scale each row to length 1, so a dot product is a cosine similarity."""
    m = np.asarray(m, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def fingerprint(model_name, wine_ids, texts):
    """A short hash that changes if the model, the wines or any note changes."""
    h = hashlib.sha256(model_name.encode())
    for wine_id, text in zip(wine_ids, texts):
        h.update(f"{wine_id}\t{text}\n".encode())
    return h.hexdigest()


class SemanticRecommender(Recommender):
    name = "semantic"

    def __init__(self, model_name, revision=None, batch_size=64, cache_dir=None, encoder=None):
        """encoder: anything with .encode(list_of_texts) -> array. Left as None,
        the Sentence-BERT model is loaded the first time it is needed; tests
        pass a small fake encoder instead, so they need no download."""
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._encoder = encoder

    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # slow import, only when needed
            self._encoder = SentenceTransformer(self.model_name, revision=self.revision)
        return self._encoder

    def encode(self, texts):
        vectors = self.encoder().encode(list(texts), batch_size=self.batch_size,
                                        show_progress_bar=len(texts) > 1000)
        return unit_vectors(vectors)

    def fit(self, catalogue, holdout_ids=()):
        super().fit(catalogue, holdout_ids)
        self.embeddings = self.load_or_encode()
        return self

    def cache_path(self):
        return repo_path(self.cache_dir) / f"{self.model_name.split('/')[-1]}.npz"

    def load_or_encode(self):
        ids = self.catalogue["wine_id"].to_numpy()
        texts = self.catalogue["description"].tolist()
        fp = fingerprint(self.model_name, ids, texts)

        if self.cache_dir is not None and self.cache_path().exists():
            cached = np.load(self.cache_path())
            if str(cached["fingerprint"]) == fp:
                log.info("loaded embeddings from %s", self.cache_path())
                return cached["embeddings"]
            log.info("cache is out of date, re-encoding")

        log.info("encoding %d notes with %s", len(texts), self.model_name)
        embeddings = self.encode(texts)
        if self.cache_dir is not None:
            self.cache_path().parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path(), embeddings=embeddings, wine_ids=ids, fingerprint=np.array(fp))
        return embeddings

    def score(self, query):
        return self.embeddings @ self.encode([query])[0]

    def vectors(self):
        """wine_id -> embedding, for the intra-list diversity metric."""
        return dict(zip(self.catalogue["wine_id"], self.embeddings))


def from_config(cfg):
    s = cfg["semantic"]
    return SemanticRecommender(s["model"], revision=s["revision"],
                               batch_size=s["batch_size"], cache_dir=s["cache_dir"])


def main():
    """`make embed`: build (or check) the embedding cache."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    wines = pd.read_parquet(repo_path(cfg["paths"]["processed_dir"]) / "wines.parquet")
    model = from_config(cfg).fit(wines)
    log.info("embeddings: %s, cache: %s", model.embeddings.shape, model.cache_path())


if __name__ == "__main__":
    main()
