import json

import pandas as pd
import pytest

from sensoryrec.judge import (agreement, describe, judge_pairwise, judge_sets, pairwise_summary, parse_pairwise,
                              parse_scores, random_lists, sample_tiers, summarise, tier)
from sensoryrec.llm import CachedLLM


def row(**wishes):
    return pd.Series({"kind": "a red wine", **wishes})


def test_tiers():
    assert tier(row(flavours="cherry")) == "basic"
    assert tier(row(flavours="cherry", budget="cheap")) == "basic"
    assert tier(row(flavours="cherry", budget="cheap", weight="light")) == "medium"
    assert tier(row(flavours="a", budget="b", weight="c", feel="d")) == "medium"
    assert tier(row(flavours="a", budget="b", weight="c", feel="d", country="e")) is None   # 5 wishes, no tier
    assert tier(row(occasion="a picnic", flavours="a", budget="b")) == "complex"
    assert tier(row(occasion="a picnic", flavours="a")) is None                             # occasion + only 1
    assert tier(row()) is None                                                              # nothing but the kind
    assert tier(row(flavours=float("nan"), budget="cheap")) == "basic"                      # empty cells do not count


def queries(n=12):
    rows = []
    for i in range(n):
        base = {"query_id": i, "query": f"q{i}", "flagged": i == 0, "kind": "a red wine"}
        if i % 3 == 0:
            base.update(flavours="cherry")
        elif i % 3 == 1:
            base.update(flavours="a", budget="b", weight="c")
        else:
            base.update(occasion="a picnic", flavours="a", budget="b")
        rows.append(base)
    return pd.DataFrame(rows)


def test_sample_tiers_is_seeded_and_skips_flagged():
    s = sample_tiers(queries(), 3, seed=42)
    assert s.groupby("tier", observed=True).size().tolist() == [3, 3, 3]
    assert 0 not in s["query_id"].tolist()                  # flagged
    assert s.equals(sample_tiers(queries(), 3, seed=42))
    with pytest.raises(ValueError, match="fewer than"):
        sample_tiers(queries(), 10, seed=42)


def test_random_lists():
    a = random_lists([1, 2], range(100), 5, seed=1)
    assert a == random_lists([1, 2], range(100), 5, seed=1)
    assert all(len(set(v)) == 5 for v in a.values())


def answer(scores, set_score, occasion=None):
    data = {"wines": [{"reason": "x", "relevance": s} for s in scores], "set_relevance": set_score}
    if occasion is not None:
        data["occasion_fit"] = occasion
    return json.dumps(data)


def test_parse_scores():
    assert parse_scores(answer([5, 4, 3], 4), 3, False) == ([5, 4, 3], 4, None)
    assert parse_scores(answer([1, 1, 1], 2, 3), 3, True) == ([1, 1, 1], 2, 3)
    with pytest.raises(ValueError):
        parse_scores(answer([5, 4], 4), 3, False)            # one wine missing
    with pytest.raises(ValueError):
        parse_scores(answer([6, 4, 3], 4), 3, False)         # out of range
    with pytest.raises(KeyError):
        parse_scores(answer([5, 4, 3], 4), 3, True)          # occasion missing
    with pytest.raises(ValueError):
        parse_scores("not json", 3, False)


def test_parse_pairwise():
    assert parse_pairwise('{"reason": "r", "better": "B"}') == "B"
    with pytest.raises(ValueError):
        parse_pairwise('{"better": "C"}')


WINES = pd.DataFrame({"wine_id": [1, 2, 3], "variety": ["Merlot", "Riesling", "Syrah"],
                      "country": ["France", "Germany", "Chile"], "price": [10.0, None, 20.0],
                      "description": ["soft red", "sweet white", "spicy red"]}).set_index("wine_id")


def test_describe_hides_names():
    text = describe(WINES, [2, 1])
    assert text.splitlines()[0] == 'Wine 1: Riesling, Germany, price unknown. Tasting note: "sweet white"'
    assert "Wine 2: Merlot, France, $10" in text


def fake_llm(tmp_path, reply):
    seen = []

    def call(prompt):
        seen.append(prompt)
        return reply(prompt)
    return CachedLLM("judge", {"temperature": 0, "seed": 42}, tmp_path, url=None, digest="d", call=call), seen


PROMPT = "R: {request}\n{wines}\nk={k}\n{occasion_rule}{occasion_field}"


def test_judge_sets(tmp_path):
    sample = pd.DataFrame({"query_id": [7, 8], "tier": ["basic", "complex"], "query": ["a red", "a red for a picnic"]})
    lists = {"m1": {7: [1, 3], 8: [3, 1]}, "m2": {7: [2, 1], 8: [2, 3]}}

    def reply(prompt):
        if "picnic" in prompt:
            return answer([4, 2], 3, occasion=5) if "Syrah" in prompt.split("\n")[1] else "broken"
        return answer([5, 1], 4)
    llm, seen = fake_llm(tmp_path, reply)
    items, sets = judge_sets(sample, lists, WINES, llm, PROMPT, k=2)
    assert len(seen) == 4                                                  # 2 requests x 2 models
    assert "occasion_fit" in seen[2] and "occasion_fit" not in seen[0]     # only complex asks for it (request 8 comes third)
    s = sets.set_index(["query_id", "model"])
    assert s.loc[(8, "m1"), "occasion_fit"] == 5 and pd.isna(s.loc[(7, "m1"), "occasion_fit"])
    assert not s.loc[(8, "m2"), "parsed"]                                  # broken answer kept as unusable
    assert len(items) == 6                                                 # 3 usable sets x 2 wines
    assert items[(items.query_id == 7) & (items.model == "m1")]["wine_id"].tolist() == [1, 3]


def test_judge_pairwise_maps_the_shown_order_back(tmp_path):
    sample = pd.DataFrame({"query_id": range(20), "tier": ["basic"] * 20, "query": ["q"] * 20})
    a = {q: [1] for q in range(20)}
    b = {q: [2] for q in range(20)}
    # The judge always prefers the list that contains the Merlot (wine 1, model "x")
    llm, _ = fake_llm(tmp_path, lambda p: '{"better": "A"}' if p.index("Merlot") < p.index("Riesling") else '{"better": "B"}')
    pw = judge_pairwise(sample, a, b, ["x", "y"], WINES, llm, "{request}\nA:{list_a}\nB:{list_b}", seed=3)
    assert (pw["winner"] == "x").all()                       # whichever position it was shown in
    assert set(pw["shown_first"]) == {"x", "y"}              # both orders occurred
    summary = pairwise_summary(pw, ["x", "y"]).set_index("tier")
    assert summary.loc["all", "pct_x_wins"] == 100


def test_summarise_and_agreement():
    items = pd.DataFrame({"model": ["m"] * 4, "tier": ["basic", "basic", "medium", "medium"],
                          "relevance": [5, 3, 4, 2]})
    sets = pd.DataFrame({"model": ["m", "m"], "tier": ["basic", "medium"], "set_relevance": [4, 2],
                         "occasion_fit": [None, None], "parsed": [True, True]})
    s = summarise(items, sets).set_index("tier")
    assert s.loc["basic", "item_relevance"] == 4 and s.loc["basic", "pct_relevant"] == 50
    assert s.loc["all", "item_relevance"] == 3.5 and s.loc["all", "set_relevance"] == 3
    human = pd.DataFrame({"query_id": [1, 2, 3], "model": ["m"] * 3, "human": [4, 2, 5]})
    judge = pd.DataFrame({"query_id": [1, 2, 3], "model": ["m"] * 3, "judge": [4, 3, 3]})
    a = agreement(human, judge)
    assert a["n"] == 3 and a["pct_exact"] == pytest.approx(100 / 3) and a["pct_within_1"] == pytest.approx(200 / 3)


def test_validation_sheet_hides_the_model():
    from sensoryrec.judge import split_validation
    sample = pd.DataFrame({"case": [1, 2], "query_id": [7, 8], "model": ["semantic", "random"],
                           "request": ["a", "b"], "wines": ["x", "y"], "your_set_relevance_1_to_5": ["", ""]})
    sheet, key = split_validation(sample)
    assert "model" not in sheet.columns and list(key.columns) == ["case", "query_id", "model"]
    assert key["model"].tolist() == ["semantic", "random"]
