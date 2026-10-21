"""TF-IDF recommender: keyword matching between the query and tasting notes.

Each note becomes a vector with one entry per word in the vocabulary:
  TF  = how often the word appears in the note (dampened: 1 + log(count)),
  IDF = how rare the word is across notes (rare words weigh more).
The query is turned into a vector the same way, and wines are ranked by
cosine similarity. Vectors have length 1, so cosine is a dot product, as
in the Semantic model.

It only matches words that literally appear in both texts: "not too heavy"
shares no word with "light-bodied". That is the baseline Semantic must beat.

The vocabulary and IDF weights are learned on the catalogue minus the test
wines (CLAUDE.md rule 4); the test wines are still turned into vectors,
because they are the targets.
"""
from sklearn.feature_extraction.text import TfidfVectorizer

from sensoryrec.recommenders.base import Recommender


class TfidfRecommender(Recommender):
    name = "tfidf"

    def __init__(self, min_df=2, stop_words="english", sublinear_tf=True, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(min_df=min_df, stop_words=stop_words,
                                          sublinear_tf=sublinear_tf, ngram_range=tuple(ngram_range))

    def fit(self, catalogue, holdout_ids=()):
        super().fit(catalogue, holdout_ids)
        seen = self.catalogue[~self.catalogue["wine_id"].isin(self.holdout_ids)]
        self.vectorizer.fit(seen["description"])
        self.vectors = self.vectorizer.transform(self.catalogue["description"])
        return self

    def score(self, query):
        q = self.vectorizer.transform([query])
        return (self.vectors @ q.T).toarray().ravel()


def from_config(cfg):
    t = cfg["tfidf"]
    return TfidfRecommender(min_df=t["min_df"], stop_words=t["stop_words"],
                            sublinear_tf=t["sublinear_tf"], ngram_range=t["ngram_range"])
