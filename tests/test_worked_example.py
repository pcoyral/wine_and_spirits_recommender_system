import json

import numpy as np
import pandas as pd

from sensoryrec import worked_example as we
from sensoryrec.llm import CachedLLM
from sensoryrec.recommenders.base import rank

CAT = pd.DataFrame({
    "wine_id": [1, 2, 3], "title": ["Dry One", "Sweet Two", "Fruity Three"],
    "variety": ["Merlot", "Riesling", "Gamay"], "country": ["US", "Germany", "France"],
    "price": [12.0, 18.0, 30.0], "points": [86, 88, 90],
    "description": ["Dry and firm with black cherry.", "Sweet ripe peach and honey.", "Fruity red berries, light."],
})


class FakeModel:
    query_prefix = ""
    catalogue = CAT

    def recommend(self, query, k=5, filters=None):
        return rank(CAT, [3.0, 1.0, 2.0], k, filters)            # Dry One first

    def encode(self, texts):
        return np.zeros((len(texts), 2))


class FakeRefiner:
    directions = {"sweetness": np.array([1.0, 0.0])}

    def scores(self, q):
        return np.array([1.0, 3.0, 2.0])                          # sweeter: Sweet Two first


def llm(tmp_path):
    def reply(prompt):
        n = sum(f"Wine {i}:" in prompt for i in range(1, 6))
        return json.dumps({"intro": "Sweeter now." if "sweeter" in prompt else "First pick.",
                           "wines": [{"wine": i, "why": f'Because "of the note" {i}.'} for i in range(1, n + 1)]})
    return CachedLLM("fake", {}, tmp_path, "", digest="fake", call=reply)


EX = {"persona": "Lucía, 24", "request": "fruity, not too dry", "max_price": 20,
      "refinement": {"label": "sweeter", "trait": "sweetness", "sign": 1}}


def test_two_turns_respect_budget_and_refinement(tmp_path):
    turns = we.run(FakeModel(), FakeRefiner(), 0.1, llm(tmp_path), "{request}{change}{k}{wines}", EX, k=2)
    first, second = turns[0][1], turns[1][1]
    assert first["wine_id"].tolist() == [1, 2]                     # $30 wine filtered out
    assert second["wine_id"].tolist() == [2, 1]                    # refinement reordered
    assert turns[0][2]["intro"] == "First pick." and turns[1][2]["intro"] == "Sweeter now."


def test_pages_show_persona_wines_and_explanations(tmp_path):
    turns = we.run(FakeModel(), FakeRefiner(), 0.1, llm(tmp_path), "{request}{change}{k}{wines}", EX, k=2)
    md = we.to_markdown(EX, turns, "semantic")
    assert "Lucía, 24" in md and "Sweet Two" in md and "Why:" in md and "at most $20" in md
    page = we.to_html(EX, turns, "semantic", {"Merlot": "red", "Riesling": "white"}, {}, {})
    assert page.startswith("<!doctype html>") and "<svg" in page and "Sweeter now." in page
