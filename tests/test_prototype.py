import numpy as np
import pandas as pd
import pytest

from sensoryrec.prototype import cheaper_limit, chosen_alpha, recommend, refinement_buttons, shared_words
from sensoryrec.recommenders.semantic import SemanticRecommender
from sensoryrec.refinement import Refiner

LEX = {"sweetness": {"more": "sweeter", "less": "drier", "build": ["sweet"], "measure": ["honey"]}}


def test_refinement_buttons():
    assert refinement_buttons(LEX) == [("sweeter", "sweetness", 1), ("drier", "sweetness", -1)]


def test_chosen_alpha(tmp_path):
    assert chosen_alpha(tmp_path / "missing.csv", default=0.05) == (0.05, False)
    pd.DataFrame({"alpha": [0.05, 0.1], "chosen": [False, True]}).to_csv(tmp_path / "a.csv", index=False)
    assert chosen_alpha(tmp_path / "a.csv", default=0.05) == (0.1, True)


def test_cheaper_limit():
    assert cheaper_limit(pd.DataFrame({"price": [10.0, None, 30.0, 20.0]})) == 20.0
    assert cheaper_limit(pd.DataFrame({"price": [None, None]})) is None


def test_shared_words():
    assert shared_words("A fruity red with cherry", "Bright cherry fruit, a fruity red wine.") == \
        ["cherry", "fruity", "red"]   # "wine" and stop words never count
    assert shared_words("isn't too heavy", "isn't heavy") == ["heavy"]   # no "isn" stub


class Words:
    WORDS = ["red", "sweet", "dry"]

    def encode(self, texts, **kwargs):
        return np.array([[t.split().count(w) for w in self.WORDS] for t in texts], dtype=float) + 1e-3


def catalogue():
    return pd.DataFrame({"wine_id": [1, 2, 3], "price": [10.0, 20.0, 30.0],
                         "country": ["US", "US", "France"], "variety": ["A", "A", "B"],
                         "description": ["red dry", "red sweet", "red"]})


def test_recommend_without_steps_is_the_model():
    model = SemanticRecommender("fake", encoder=Words()).fit(catalogue())
    assert recommend(model, "red", 2, None)["wine_id"].tolist() == model.recommend("red", k=2)["wine_id"].tolist()


def test_sweeter_moves_towards_the_sweet_note():
    model = SemanticRecommender("fake", encoder=Words()).fit(catalogue())
    refiner = Refiner(model.embeddings, model.catalogue, LEX).fit(holdout_ids=[])
    before = recommend(model, "red", 1, None)["wine_id"].tolist()
    after = recommend(model, "red", 1, None, steps=[("sweetness", 1)], refiner=refiner, alpha=0.5)["wine_id"].tolist()
    drier = recommend(model, "red", 3, None, steps=[("sweetness", -1)], refiner=refiner, alpha=0.5)["wine_id"].tolist()
    assert before == [3] and after == [2]
    assert drier[-1] == 2   # "drier" pushes the sweet note to the bottom


def test_steps_need_a_refiner():
    model = SemanticRecommender("fake", encoder=Words()).fit(catalogue())
    with pytest.raises(ValueError, match="Refiner"):
        recommend(model, "red", 1, None, steps=[("sweetness", 1)])
