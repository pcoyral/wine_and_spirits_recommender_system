import pytest

from sensoryrec.config import repo_path
from sensoryrec.text import find_terms, overlap_share
from sensoryrec.vocab import expert_terms_used, leaked_names, load_vocab, violations

VOCAB = load_vocab(repo_path("data/wine_vocab.txt"))


def test_vocab_file_loads_both_tiers():
    assert "tannin" in VOCAB["expert"] and "sweet" in VOCAB["everyday"]
    # the decided tiers: oaky / tannic / acidic banned, colour words allowed
    assert {"oaky", "tannic", "acidic"} <= set(VOCAB["expert"])
    assert {"red", "white", "sparkling"} <= set(VOCAB["everyday"])
    # the two tiers never contradict each other
    assert not set(VOCAB["expert"]) & set(VOCAB["everyday"])


def test_everyday_words_pass():
    assert expert_terms_used("A sweet, fruity red that feels light and smooth.", VOCAB) == []


@pytest.mark.parametrize("query, banned", [
    ("Soft tannins and a long finish", ["tannin", "finish"]),          # plural counts
    ("A Full-Bodied red", ["bodied"]),                                  # case, hyphen
    ("tastes of stone fruit", ["stone fruit"]),                         # multi-word
    ("a lovely cuvee", ["cuvée"]),                                      # accents ignored
])
def test_expert_terms_are_caught(query, banned):
    assert expert_terms_used(query, VOCAB) == banned


def test_whole_words_only():
    # "oaky" is banned, but "soaky" or "nosey" are other words
    assert find_terms("soaky and nosey", ["oaky", "nose"]) == []


def test_negation_guard():
    assert find_terms("not sweet at all", ["sweet"], negation_guard=True) == []
    assert find_terms("not sweet at all", ["sweet"]) == ["sweet"]
    assert find_terms("sweet, not dry", ["sweet"], negation_guard=True) == ["sweet"]


WINE = {"variety": "Pinot Noir", "winery": "Domaine X", "region_1": "Sancerre",
        "region_2": None, "province": "Loire Valley", "country": "France", "designation": "Unknown"}


def test_leak_check_finds_own_names():
    assert leaked_names("Like a light french pinot noir from sancerre", WINE) == ["Pinot Noir", "Sancerre"]
    # "french" is an adjective, not the name "France"; only names count
    assert leaked_names("A light, cherry red", WINE) == []


def test_everyday_names_are_not_leaks():
    # A wine whose variety is "Rosé" may still be called "rosé" (D7);
    # the country "US" is not the pronoun "us".
    rose = {"variety": "Rosé", "country": "US", "winery": "Y", "designation": None}
    assert leaked_names("a dry rosé for us", rose, everyday=VOCAB["everyday"]) == []


def test_violations_combines_both_checks():
    assert violations("Pinot noir with soft tannins", WINE, VOCAB) == ["tannin", "Pinot Noir"]


def test_overlap_share():
    # content words of the query: {bright, cherry, smooth} ("a", "and" are
    # stop words, "wine" is too generic); the note has "cherry" and "smooth" -> 2 of 3
    note = "Ripe cherry flavors, smooth and round."
    assert overlap_share("A bright, cherry and smooth wine", note) == pytest.approx(2 / 3)
    assert overlap_share("", note) == 0.0
