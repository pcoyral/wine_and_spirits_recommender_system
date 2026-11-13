import pandas as pd
import pytest

from sensoryrec.config import repo_path
from sensoryrec.llm import CachedLLM, gemini_call, gemini_from_config, gemini_text
from sensoryrec.simulate import build_briefs, quality, simulate, wishes_text
from sensoryrec.vocab import load_vocab

VOCAB = load_vocab(repo_path("data/wine_vocab.txt"))
FACETS = {
    "experience": ["knows nothing about wine", "drinks wine now and then"],
    "kind": ["a red wine", "a sweet wine for dessert"],
    "optional": {
        "sweetness": {"chance": 1.0, "values": ["not sweet at all"]},
        "flavours": {"chance": 1.0, "pick": 2, "values": ["cherry", "plum", "lemon"]},
        "occasion": {"chance": 0.5, "values": ["a picnic", "a barbecue"]},
    },
}
PROMPT = "The customer {experience}. They want:\n{wishes}\nBanned: {banned}\n"
RETRY = "\nYour previous answer was: {previous}\nNot allowed: {problems}\n"


def llm(tmp_path, answers):
    """A cached LLM whose 'model' answers from a function of the prompt."""
    seen = []

    def call(prompt):
        seen.append(prompt)
        return answers(prompt)
    return CachedLLM("fake", {"temperature": 0, "seed": 42}, tmp_path, url=None, digest="d1", call=call), seen


def test_briefs_are_distinct_and_repeatable():
    briefs = build_briefs(FACETS, 20, seed=42)
    assert len(briefs) == 20
    assert len({tuple(sorted(b.items())) for b in briefs}) == 20  # no brief twice
    assert briefs == build_briefs(FACETS, 20, seed=42)            # same seed, same briefs
    assert briefs != build_briefs(FACETS, 20, seed=7)


def test_brief_contents():
    for brief in build_briefs(FACETS, 20, seed=42):
        assert brief["experience"] in FACETS["experience"] and brief["kind"] in FACETS["kind"]
        # chance 1.0 -> always there; one or two different flavours
        flavours = brief["flavours"].split(" and ")
        assert 1 <= len(flavours) <= 2 and len(set(flavours)) == len(flavours)
        # a sweet wine that is "not sweet at all" would contradict itself
        assert ("sweetness" in brief) == (brief["kind"] == "a red wine")


def test_too_few_possible_briefs_is_an_error():
    tiny = {"experience": ["a"], "kind": ["b"], "optional": {}}  # only one brief exists
    with pytest.raises(ValueError, match="only allow 1"):
        build_briefs(tiny, 2, seed=42)


def test_wishes_text_leaves_out_who_is_asking():
    brief = {"experience": "knows nothing about wine", "kind": "a red wine", "occasion": "a picnic"}
    assert wishes_text(brief) == "- kind: a red wine\n- occasion: a picnic"


def test_simulate_retries_on_expert_words_and_flags(tmp_path):
    briefs = [{"experience": "knows nothing about wine", "kind": "a red wine"},
              {"experience": "knows nothing about wine", "kind": "a rosé"}]

    def answers(prompt):
        if "a rosé" in prompt:
            return "Something with tannins please"               # never fixed -> flagged
        return "A smooth red please" if "previous answer" in prompt else "A red with nice tannins"

    model, seen = llm(tmp_path, answers)
    out = simulate(briefs, model, VOCAB, PROMPT, RETRY, max_retries=3)

    assert out["query_id"].tolist() == [1, 2]
    first, second = out.iloc[0], out.iloc[1]
    # Brief 1: the expert word is named in the retry prompt, the rewrite is clean
    assert first["query"] == "A smooth red please" and first["attempts"] == 2
    assert first["first_expert"] == "tannin" and not first["flagged"]
    assert "Not allowed: tannin" in seen[1]
    # Brief 2: the same answer again stops the retries; the query is flagged
    assert second["flagged"] and second["violations"] == "tannin" and second["attempts"] == 1
    # The brief is kept next to the query
    assert first["kind"] == "a red wine" and second["kind"] == "a rosé"


def test_empty_answer_is_flagged(tmp_path):
    model, _ = llm(tmp_path, lambda prompt: "")
    out = simulate([{"experience": "x", "kind": "a red wine"}], model, VOCAB, PROMPT, RETRY, max_retries=3)
    assert out.loc[0, "flagged"] and out.loc[0, "violations"] == "an empty answer"


def test_quality_row():
    queries = pd.DataFrame({"query": ["a b c", "A B C", "d e"], "attempts": [1, 2, 1],
                            "first_expert": ["", "tannins", None], "flagged": [False, False, True]})
    row = quality(queries)
    assert row["n_queries"] == 3 and row["n_distinct_queries"] == 2  # "a b c" twice, ignoring case
    assert row["pct_expert_first"] == pytest.approx(100 / 3)
    assert row["pct_regenerated"] == pytest.approx(100 / 3)
    assert row["pct_flagged"] == pytest.approx(100 / 3)
    assert row["words_min"] == 2 and row["words_max"] == 3


def test_gemini_text():
    answer = {"candidates": [{"content": {"parts": [{"text": "A light "}, {"text": "red please"}]}}]}
    assert gemini_text(answer) == "A light red please"
    # no text at all (e.g. a blocked prompt) -> "", which simulate() flags
    assert gemini_text({"promptFeedback": {"blockReason": "OTHER"}}) == ""
    assert gemini_text({"candidates": [{"finishReason": "SAFETY"}]}) == ""


def gemini_cfg(tmp_path, model):
    return {"llm": {"cache_dir": tmp_path},
            "simulation": {"llm": {"model": model, "url": "https://example.invalid/{model}",
                                   "temperature": 0, "seed": 42, "pause_seconds": 0}}}


def test_gemini_needs_a_pinned_model_name(tmp_path):
    with pytest.raises(SystemExit, match="simulation.llm.model"):
        gemini_from_config(gemini_cfg(tmp_path, None))


def test_gemini_needs_the_key_in_the_environment(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    call = gemini_call("some-model", "https://example.invalid/{model}", {"temperature": 0, "seed": 42})
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):   # stops before any request is sent
        call("hello")


def test_gemini_answers_are_cached_under_the_model_name(tmp_path):
    asked = []
    model = gemini_from_config(gemini_cfg(tmp_path, "model-a"), call=lambda p: asked.append(p) or "hi")
    assert model.generate("hello") == "hi" and model.generate("hello") == "hi"
    assert asked == ["hello"]                                  # the second time comes from the cache
    other = gemini_from_config(gemini_cfg(tmp_path, "model-b"), call=lambda p: "other")
    assert other.generate("hello") == "other"                  # another model, another cache entry
