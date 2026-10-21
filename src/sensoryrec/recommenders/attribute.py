"""Attribute recommender: reads structured wishes out of the query and ranks
wines by how many of them they satisfy.

A simple rule-based reader (no LLM, so ranking stays deterministic, CLAUDE.md
rule 2) looks for four kinds of attribute in the query:
- variety:  a grape name from the catalogue, e.g. "pinot noir"
- country:  a country name or adjective, e.g. "italy" / "italian"
- colour:   red / white / rosé / sparkling, and everyday words for them
            ("pink", "bubbly"). The dataset has no colour column, so a wine's
            colour comes from data/variety_colour.csv, a hand-made table for
            the most common varieties (DECISIONS.md).
- price:    explicit amounts only: "under 20", "less than $30",
            "between 10 and 20". The number is compared with the dataset's
            prices (US dollars) without currency conversion.

Score = number of attributes asked for that the wine matches, with points as
tie-breaker (points / 1000, as in Popularity). It is a soft score, not a
filter, so the model always returns k wines. If the query mentions no
attribute at all, every wine scores 0 and the ranking is simply by points.

Variety and country lists are learned from the catalogue minus the test
wines (CLAUDE.md rule 4).
"""
import re

import numpy as np
import pandas as pd

from sensoryrec.config import repo_path
from sensoryrec.recommenders.base import Recommender
from sensoryrec.text import normalise

# Everyday words for the four colours (after removing accents: rosé -> rose)
COLOUR_WORDS = {
    "red": "red", "reds": "red",
    "white": "white", "whites": "white",
    "rose": "rosé", "pink": "rosé", "blush": "rosé",
    "sparkling": "sparkling", "bubbly": "sparkling", "bubbles": "sparkling",
    "fizzy": "sparkling", "champagne": "sparkling", "prosecco": "sparkling", "cava": "sparkling",
}

# Adjectives and other names for countries in the dataset
COUNTRY_WORDS = {
    "american": "US", "america": "US", "usa": "US",
    "argentinian": "Argentina", "argentine": "Argentina", "australian": "Australia",
    "austrian": "Austria", "canadian": "Canada", "chilean": "Chile", "french": "France",
    "german": "Germany", "greek": "Greece", "hungarian": "Hungary", "israeli": "Israel",
    "italian": "Italy", "portuguese": "Portugal", "south african": "South Africa",
    "spanish": "Spain", "uruguayan": "Uruguay",
}

# Names that are also everyday words; never read as a variety or country
# ("melon flavours" is not a request for the Melon grape; "us" is a pronoun).
IGNORED_NAMES = {"melon", "apple", "other", "mission", "debit", "vital", "us"}

PRICE_MAX = re.compile(r"\b(?:under|below|less than|cheaper than|up to|no more than|at most|max(?:imum)?)"
                       r"\s*[$€£]?\s*(\d+(?:\.\d+)?)")
# (?<!no ) so that "no more than 20" is read only as a maximum
PRICE_MIN = re.compile(r"(?<!no )\b(?:over|above|more than|at least|min(?:imum)?)\s*[$€£]?\s*(\d+(?:\.\d+)?)")
PRICE_OR_LESS = re.compile(r"[$€£]?\s*(\d+(?:\.\d+)?)\s*(?:euros?|dollars?|bucks|eur|usd)?\s*or (?:less|under|cheaper)")
PRICE_BETWEEN = re.compile(r"(?:between|from)\s*[$€£]?\s*(\d+(?:\.\d+)?)\s*(?:and|to)\s*[$€£]?\s*(\d+(?:\.\d+)?)")


def find_names(text, names):
    """Which names (normalised phrase -> value) appear in the normalised text
    as whole words. Longest names first, and a matched phrase is removed, so
    "sauvignon blanc" is not also read as "sauvignon"."""
    padded = f" {text} "
    found = set()
    for phrase in sorted(names, key=len, reverse=True):
        if f" {phrase} " in padded:
            found.add(names[phrase])
            padded = padded.replace(f" {phrase} ", " | ")
    return found


def parse_price(query):
    """(min_price, max_price) from explicit amounts in the query; None if absent."""
    text = query.lower()
    if m := PRICE_BETWEEN.search(text):
        low, high = sorted([float(m.group(1)), float(m.group(2))])
        return low, high
    high = PRICE_MAX.search(text) or PRICE_OR_LESS.search(text)
    low = PRICE_MIN.search(text)
    return (float(low.group(1)) if low else None), (float(high.group(1)) if high else None)


class AttributeRecommender(Recommender):
    name = "attribute"

    def __init__(self, colour_table):
        """colour_table: DataFrame with columns variety, colour."""
        self.variety_colour = dict(zip(colour_table["variety"], colour_table["colour"]))

    def fit(self, catalogue, holdout_ids=()):
        super().fit(catalogue, holdout_ids)
        seen = self.catalogue[~self.catalogue["wine_id"].isin(self.holdout_ids)]
        self.variety_names = {normalise(v): v for v in seen["variety"].unique()
                              if normalise(v) not in IGNORED_NAMES}
        countries = {c for c in seen["country"].unique() if c != "Unknown"}
        self.country_names = {normalise(c): c for c in countries if normalise(c) not in IGNORED_NAMES}
        self.country_names.update({w: c for w, c in COUNTRY_WORDS.items() if c in countries})

        self.colour = self.catalogue["variety"].map(self.variety_colour).to_numpy()
        self.points = self.catalogue["points"].to_numpy()
        return self

    def parse(self, query):
        """The attributes the query asks for."""
        text = normalise(query)
        min_price, max_price = parse_price(query)
        return {
            "varieties": find_names(text, self.variety_names),
            "countries": find_names(text, self.country_names),
            "colours": {COLOUR_WORDS[w] for w in text.split() if w in COLOUR_WORDS},
            "min_price": min_price,
            "max_price": max_price,
        }

    def score(self, query):
        wish = self.parse(query)
        c = self.catalogue
        matched = np.zeros(len(c))
        if wish["varieties"]:
            matched += c["variety"].isin(wish["varieties"]).to_numpy()
        if wish["countries"]:
            matched += c["country"].isin(wish["countries"]).to_numpy()
        if wish["colours"]:
            matched += np.isin(self.colour, list(wish["colours"]))
        if wish["min_price"] is not None or wish["max_price"] is not None:
            low = wish["min_price"] if wish["min_price"] is not None else -np.inf
            high = wish["max_price"] if wish["max_price"] is not None else np.inf
            # a wine without a price never matches a price wish (unknown, not cheap)
            matched += c["price"].between(low, high).fillna(False).to_numpy()
        return matched + self.points / 1000


def from_config(cfg):
    table = pd.read_csv(repo_path(cfg["attribute"]["colour_table"]))
    return AttributeRecommender(table)
