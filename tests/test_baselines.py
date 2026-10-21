import numpy as np
import pandas as pd
import pytest

from sensoryrec.recommenders.attribute import AttributeRecommender, parse_price
from sensoryrec.text import normalise
from sensoryrec.recommenders.tfidf import TfidfRecommender


def catalogue():
    return pd.DataFrame({
        "wine_id": [1, 2, 3, 4, 5, 6],
        "winery": ["A", "A", "B", "B", "C", "D"],
        "variety": ["Pinot Noir", "Sauvignon Blanc", "Sauvignon", "Melon", "Rosé", "Gewürztraminer"],
        "country": ["Italy", "France", "France", "US", "Spain", "Germany"],
        "points": [90, 88, 85, 92, 87, 91],
        "price": [15.0, 25.0, 12.0, np.nan, 18.0, 40.0],
        "description": ["cherry and oak", "lemon and grass", "lemon",
                        "zesty melon", "strawberry", "lychee and cherry"],
    })


COLOURS = pd.DataFrame({
    "variety": ["Pinot Noir", "Sauvignon Blanc", "Sauvignon", "Melon", "Rosé", "Gewürztraminer"],
    "colour": ["red", "white", "white", "white", "rosé", "white"],
})


# ---------- TF-IDF ----------

def test_tfidf_finds_matching_words():
    model = TfidfRecommender(min_df=1).fit(catalogue())
    assert model.recommend("oak", k=1)["wine_id"].tolist() == [1]
    # "lemon" is in notes 2 and 3. Note 3 is only "lemon" (cosine 1); note 2
    # also has "grass", so its vector points partly elsewhere (cosine < 1).
    # ("and" is a stop word and is dropped.)
    assert model.recommend("lemon", k=2)["wine_id"].tolist() == [3, 2]


def test_tfidf_vocabulary_ignores_holdout_wines():
    # "lychee" appears only in wine 6. Held out, the word is not in the
    # vocabulary, so the query matches nothing (all scores 0).
    model = TfidfRecommender(min_df=1).fit(catalogue(), holdout_ids=[6])
    assert "lychee" not in model.vectorizer.vocabulary_
    assert (model.score("lychee") == 0).all()
    # Wine 6 is still in the catalogue, with a vector made of the words the
    # vocabulary knows: only "cherry". So for "cherry" it is a perfect match
    # (cosine 1) and beats wine 1 ("cherry and oak").
    assert model.recommend("cherry", k=1)["wine_id"].tolist() == [6]


def test_tfidf_keeps_meaningful_stop_words():
    # With the plain English list, "not too heavy" would become just "heavy".
    model = TfidfRecommender(min_df=1, keep_words=["not", "too", "full"]).fit(catalogue())
    analyse = model.vectorizer.build_analyzer()
    assert analyse("not too heavy, full bodied") == ["not", "too", "heavy", "full", "bodied"]
    plain = TfidfRecommender(min_df=1).fit(catalogue())
    assert plain.vectorizer.build_analyzer()("not too heavy") == ["heavy"]


def test_tfidf_has_no_idea_of_meaning():
    # The baseline weakness: no shared word, no match, whatever the meaning.
    model = TfidfRecommender(min_df=1).fit(catalogue())
    assert (model.score("citrus") == 0).all()


# ---------- Attribute: reading the query ----------

@pytest.mark.parametrize("query, expected", [
    ("something under $20", (None, 20.0)),
    ("less than 30 euros", (None, 30.0)),
    ("no more than 20", (None, 20.0)),          # not also read as "more than 20"
    ("20 euros or less", (None, 20.0)),
    ("between 30 and 10 dollars", (10.0, 30.0)),  # order does not matter
    ("over 50, a gift", (50.0, None)),
    ("a nice red", (None, None)),
])
def test_parse_price(query, expected):
    assert parse_price(query) == expected


def test_normalise():
    assert normalise("Nero d'Avola") == "nero d avola"
    assert normalise("Rosé!") == "rose"


def test_parse_finds_attributes():
    model = AttributeRecommender(COLOURS).fit(catalogue())
    wish = model.parse("An Italian Pinot Noir, something pink and bubbly")
    assert wish["varieties"] == {"Pinot Noir"}
    assert wish["countries"] == {"Italy"}
    assert wish["colours"] == {"rosé", "sparkling"}


def test_parse_longest_variety_wins():
    model = AttributeRecommender(COLOURS).fit(catalogue())
    assert model.parse("a sauvignon blanc")["varieties"] == {"Sauvignon Blanc"}
    assert model.parse("a sauvignon")["varieties"] == {"Sauvignon"}


def test_parse_ignores_everyday_words():
    model = AttributeRecommender(COLOURS).fit(catalogue())
    # "melon" is a flavour, not the Melon grape; "us" is a pronoun, not the US
    wish = model.parse("tastes like melon, for us two")
    assert wish["varieties"] == set() and wish["countries"] == set()


def test_parse_accents_do_not_matter():
    model = AttributeRecommender(COLOURS).fit(catalogue())
    assert model.parse("gewurztraminer")["varieties"] == {"Gewürztraminer"}


def test_attribute_vocabulary_ignores_holdout_wines():
    # Germany appears only in wine 6. Held out, "german" is not recognised.
    model = AttributeRecommender(COLOURS).fit(catalogue(), holdout_ids=[6])
    assert model.parse("something german")["countries"] == set()


# ---------- Attribute: ranking ----------

def test_attribute_counts_matched_wishes():
    model = AttributeRecommender(COLOURS).fit(catalogue())
    # "french white under 20": wish = country France, colour white, price <= 20
    #   wine 3: France + white + 12.0  -> 3 matches
    #   wine 2: France + white + 25.0  -> 2
    #   wine 4: white, no price        -> 1 (no price never matches a price wish)
    #   wine 6: white, 40.0            -> 1
    #   wine 1: 15.0, red, Italy       -> 1
    #   wine 5: 18.0                   -> 1
    scores = dict(zip(model.catalogue["wine_id"], model.score("french white under 20")))
    assert scores[3] == pytest.approx(3.085)  # 3 matches + 85 points / 1000
    assert scores[2] == pytest.approx(2.088)
    top = model.recommend("french white under 20", k=3)["wine_id"].tolist()
    # then the 1-match wines by points: 4 (92), 6 (91), ...
    assert top == [3, 2, 4]


def test_attribute_without_wishes_ranks_by_points():
    model = AttributeRecommender(COLOURS).fit(catalogue())
    assert model.recommend("surprise me", k=3)["wine_id"].tolist() == [4, 6, 1]
