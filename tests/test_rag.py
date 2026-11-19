import json

import pandas as pd
import pytest

from sensoryrec import rag
from sensoryrec.llm import CachedLLM

TEMPLATE = "Request: {request}\n{change}\n{k} wines:\n{wines}"


def top():
    return pd.DataFrame({
        "wine_id": [11, 12], "rank": [1, 2], "title": ["Alpha 2015 Merlot", "Beta 2018 Riesling"],
        "variety": ["Merlot", "Riesling"], "country": ["US", "Germany"], "price": [15.0, None],
        "points": [88, 90],
        "description": ["Juicy black cherry and soft tannins, easy drinking.",
                        "Off-dry, with ripe peach and a zesty lemon finish."],
    })


def answer(**over):
    data = {"intro": "Two easy wines.",
            "wines": [{"wine": 1, "why": 'It is "juicy black cherry and soft tannins".'},
                      {"wine": 2, "why": 'It is "off-dry, with ripe peach".'}]}
    data.update(over)
    return json.dumps(data)


def fake_llm(tmp_path, reply):
    return CachedLLM("fake", {"temperature": 0}, tmp_path, "", digest="fake", call=lambda prompt: reply)


def test_prompt_holds_the_ranked_wines_and_nothing_else():
    prompt = rag.build_prompt(TEMPLATE, "something fruity", top())
    assert "Wine 1: Alpha 2015 Merlot. Merlot, US, $15." in prompt
    assert "Wine 2: Beta 2018 Riesling. Riesling, Germany, price unknown." in prompt
    assert prompt.index("Alpha") < prompt.index("Beta")              # engine order kept
    assert "88" not in prompt and "11" not in prompt                  # no points, no ids


def test_parse_reads_intro_and_one_sentence_per_wine():
    out = rag.parse(answer(), 2)
    assert out["intro"] == "Two easy wines." and len(out["whys"]) == 2


@pytest.mark.parametrize("bad", [answer(wines=[{"wine": 1, "why": "x"}]), answer(intro=""), "not json"])
def test_parse_refuses_incomplete_answers(bad):
    with pytest.raises(ValueError):
        rag.parse(bad, 2)


def test_quote_check_counts_quotes_found_in_the_note():
    note = "Juicy black cherry and soft tannins, easy drinking."
    assert rag.quote_check('Has "Juicy black-cherry and soft tannins".', note) == (1, 1)
    assert rag.quote_check('Has "notes of vanilla oak" and "soft tannins".', note) == (2, 1)
    assert rag.quote_check('One word "juicy" does not count.', note) == (0, 0)


def test_explain_uses_the_llm_and_caches_the_answer(tmp_path):
    out = rag.explain(fake_llm(tmp_path, answer()), TEMPLATE, "fruity", top())
    assert out["source"] == "llm" and out["whys"][1].startswith("It is")
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_explain_falls_back_to_shared_words_when_the_llm_fails(tmp_path):
    def broken(prompt):
        raise SystemExit("quota")
    llm = CachedLLM("fake", {}, tmp_path, "", digest="fake", call=broken)
    out = rag.explain(llm, TEMPLATE, "juicy cherry please", top())
    assert out["source"] == "fallback" and "quota" in out["reason"]
    assert "cherry" in out["whys"][0] and "juicy" in out["whys"][0]


def test_explain_falls_back_on_an_unreadable_answer(tmp_path):
    assert rag.explain(fake_llm(tmp_path, "sorry"), TEMPLATE, "x", top())["source"] == "fallback"


def test_change_text_names_the_new_wines():
    text = rag.change_text("sweeter", [11, 12, 13], [12, 14, 11])
    assert '"sweeter"' in text and "Wine 2" in text and "Wine 1" not in text
    assert rag.change_text("", [1], [2]) == ""
    assert "same wines" in rag.change_text("cheaper", [1, 2], [2, 1])


class FakeModel:
    def recommend(self, query, k=5, filters=None):
        return top().head(k)


def test_faithfulness_rows_have_empty_annotation_columns(tmp_path):
    requests = pd.DataFrame({"query_id": [3, 9], "query": ["fruity red", "fresh white"]})
    sheet = rag.faithfulness_rows(requests, FakeModel(), fake_llm(tmp_path, answer()), TEMPLATE, 2)
    assert len(sheet) == 2 and set(sheet["source"]) == {"llm"}
    assert sheet["quotes"].tolist() == [2, 2] and sheet["quotes_found_in_notes"].tolist() == [2, 2]
    assert (sheet["faithful_yes_partly_no"] == "").all() and "Wine 2:" in sheet.loc[0, "explanation"]
