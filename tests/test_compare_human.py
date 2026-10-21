import pandas as pd
import pytest

from sensoryrec.compare_human import check, compare, load_human, prepare
from sensoryrec.config import repo_path
from sensoryrec.recommenders.tfidf import TfidfRecommender
from sensoryrec.vocab import load_vocab


def test_load_human_keeps_consented_queries_only(tmp_path):
    path = tmp_path / "human_queries.csv"
    pd.DataFrame({"participant_id": ["P01", "P01", "P02", "P03"], "card": [1, 2, 3, 4],
                  "wine_id": [1, 2, 3, 4], "query": ["soft red", "", "light white", "fizzy"],
                  "self_rated_knowledge": [1, 1, 2, 1], "consent": ["yes", "yes", "no", " Yes "],
                  "date": ["2026-10-30"] * 4}).to_csv(path, index=False)
    # card 2 has no query, card 3 no consent
    assert load_human(path)["wine_id"].tolist() == [1, 4]


def test_load_human_without_file_stops(tmp_path):
    with pytest.raises(SystemExit, match="No hand-written queries"):
        load_human(tmp_path / "missing.csv")


def test_check_human_queries_without_retries():
    wines = pd.DataFrame({"wine_id": [1], "variety": ["Merlot"], "winery": ["W"], "country": ["Chile"],
                          "region_1": [None], "region_2": [None], "province": ["P"],
                          "designation": [None]}).set_index("wine_id", drop=False)
    vocab = load_vocab(repo_path("data/wine_vocab.txt"))
    out = check(pd.DataFrame({"wine_id": [1], "query": ["a merlot with soft tannins"]}), wines, vocab)
    assert out.loc[0, "first_expert"] == "tannin" and out.loc[0, "first_leak"] == "Merlot"


def test_compare_counts_hits_per_source():
    catalogue = pd.DataFrame({"wine_id": [1, 2, 3], "description": ["cherry oak", "lemon grass", "honey sweet"]})
    model = TfidfRecommender(min_df=1).fit(catalogue)
    sources = {"llm": pd.DataFrame({"wine_id": [1, 3], "query": ["cherry", "honey"]}),
               "human": pd.DataFrame({"wine_id": [1, 3], "query": ["cherry", "a red one"]})}
    # k=1: LLM finds both; the human query "a red one" shares no word -> 1 hit of 2
    result = compare([model], sources, k=1).set_index("source")
    assert result.loc["llm", "hits"] == 2 and result.loc["human", "hits"] == 1
    assert result.loc["human", "hr@1"] == pytest.approx(0.5)


def test_prepare_lines_up_both_sources_on_the_same_wines():
    wines = pd.DataFrame({"wine_id": [1, 2, 3], "variety": ["Merlot"] * 3, "winery": ["W"] * 3,
                          "country": ["Chile"] * 3, "region_1": [None] * 3, "region_2": [None] * 3,
                          "province": ["P"] * 3, "designation": [None] * 3}).set_index("wine_id", drop=False)
    vocab = load_vocab(repo_path("data/wine_vocab.txt"))
    human = pd.DataFrame({"wine_id": [1, 2, 3], "query": ["soft red", "soft tannins", "light"]})
    llm_all = pd.DataFrame({"wine_id": [1, 2, 3, 9], "query": ["a", "b", "c", "d"],
                            "flagged": [False, False, True, False]})
    sources, llm_same, checked, lost = prepare(human, llm_all, wines, vocab)
    # wine 3's LLM query was flagged: lost for the HR comparison, but counted
    assert lost == {3}
    assert sources["llm"]["wine_id"].tolist() == [1, 2]
    assert sources["human_all"]["wine_id"].tolist() == [1, 2]
    # "soft tannins" uses an expert term: not in human_clean
    assert sources["human_clean"]["wine_id"].tolist() == [1]
    # quality uses ALL LLM queries of the human wines, flagged included (not wine 9)
    assert llm_same["wine_id"].tolist() == [1, 2, 3]
