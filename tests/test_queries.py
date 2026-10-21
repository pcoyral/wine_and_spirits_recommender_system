import json

import pandas as pd
import pytest

from sensoryrec.config import repo_path
from sensoryrec.llm import CachedLLM
from sensoryrec.queries import clean, quality, write_query
from sensoryrec.vocab import load_vocab

VOCAB = load_vocab(repo_path("data/wine_vocab.txt"))
PROMPT = "Note: {note}\nBanned: {banned}\n"
RETRY = "\nYour previous answer was: {previous}\nNot allowed: {problems}\n"
WINE = pd.Series({"wine_id": 7, "description": "Firm tannins and ripe cherry.", "variety": "Merlot",
                  "winery": "W", "region_1": None, "region_2": None, "province": "P",
                  "country": "Chile", "designation": None})


def llm(tmp_path, answers):
    """A CachedLLM whose 'model' answers from a function of the prompt."""
    seen = []

    def call(prompt):
        seen.append(prompt)
        return answers(prompt)
    model = CachedLLM("fake", {"temperature": 0, "seed": 42}, tmp_path, url=None, digest="d1", call=call)
    return model, seen


def test_cache_stores_and_reuses(tmp_path):
    model, seen = llm(tmp_path, lambda p: "an answer")
    assert model.generate("hello") == "an answer" and model.n_calls == 1
    # The answer is on disk with everything needed to reproduce it.
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["prompt"] == "hello" and record["options"] == {"temperature": 0, "seed": 42}
    # Same prompt again: read from the cache, the model is not asked.
    again, seen2 = llm(tmp_path, lambda p: "something else")
    assert again.generate("hello") == "an answer" and again.n_calls == 0 and seen2 == []


def test_cache_key_depends_on_model_version(tmp_path):
    a = CachedLLM("fake", {}, tmp_path, None, digest="d1", call=str)
    b = CachedLLM("fake", {}, tmp_path, None, digest="d2", call=str)
    assert a.key("x") != b.key("x")


def test_clean():
    assert clean('  "A soft,\n fruity red."  ') == "A soft, fruity red."


def test_retry_names_the_banned_words(tmp_path):
    # First answer uses "tannins"; the retry prompt must name the problem.
    answers = lambda p: "Soft and cherry-like." if "previous answer" in p else "Soft tannins and cherry."
    model, seen = llm(tmp_path, answers)
    row = write_query(WINE, model, VOCAB, PROMPT, RETRY, max_retries=3)
    assert row["query"] == "Soft and cherry-like."
    assert row["attempts"] == 2 and row["first_expert"] == "tannin" and not row["flagged"]
    assert "Not allowed: tannin" in seen[1] and "Soft tannins and cherry." in seen[1]


def test_flagged_after_max_retries(tmp_path):
    # Each retry gives a new answer that still names the grape:
    # 1 try + 3 retries, then flagged.
    count = iter(range(100))
    model, seen = llm(tmp_path, lambda p: f"A merlot, version {next(count)}")
    row = write_query(WINE, model, VOCAB, PROMPT, RETRY, max_retries=3)
    assert row["attempts"] == 4 and row["flagged"] and len(seen) == 4
    assert row["violations"] == "Merlot; number or price"


def test_repeated_answer_stops_retrying(tmp_path):
    # The model repeats the same wrong answer: a further retry prompt would be
    # identical (and so would the answer), so it stops after one retry.
    model, seen = llm(tmp_path, lambda p: "A merlot, great for $10.")
    row = write_query(WINE, model, VOCAB, PROMPT, RETRY, max_retries=3)
    assert row["attempts"] == 2 and row["flagged"] and len(seen) == 2


def test_clean_query_needs_one_attempt(tmp_path):
    model, seen = llm(tmp_path, lambda p: "A juicy, cherry-flavoured red.")
    row = write_query(WINE, model, VOCAB, PROMPT, RETRY, max_retries=3)
    assert row["attempts"] == 1 and row["first_expert"] == "" and not row["flagged"]


def test_quality_row():
    queries = pd.DataFrame({
        "wine_id": [1, 2], "query": ["soft cherry red", "bright lemon taste overall"],
        "attempts": [1, 3], "first_expert": ["", "tannin"], "first_leak": ["", ""],
        "flagged": [False, True]})
    notes = {1: "Soft cherry flavors.", 2: "Lemon and lime."}
    row = quality(queries, notes, "toy")
    assert row["pct_expert_first"] == pytest.approx(50.0)
    assert row["pct_regenerated"] == pytest.approx(50.0) and row["mean_attempts"] == pytest.approx(2.0)
    assert row["pct_flagged"] == pytest.approx(50.0)
    assert row["words_mean"] == pytest.approx(3.5) and row["words_max"] == 4
    # copy share: query 1 {soft, cherry, red} vs note {soft, cherry, flavors} -> 2/3;
    # query 2 {bright, lemon, taste, overall} vs {lemon, lime} -> 1/4; mean = 11/24
    assert row["copy_share_mean"] == pytest.approx((2 / 3 + 1 / 4) / 2)
