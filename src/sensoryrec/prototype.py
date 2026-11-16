"""Logic behind the S3 chat PROTOTYPE (app/chat.py), kept apart from the
screen code so it can be tested.

What the prototype does: a request in everyday words -> the top k of one
model -> optional refinement steps ("sweeter", "lighter", "cheaper") -> for
each wine, the words its tasting note shares with the request.

What it does not do yet: the LLM explanation grounded in the notes (S3,
rag.py) and bottle images (increment A). Ranking is deterministic, as
everywhere (CLAUDE.md rule 2).
"""
import numpy as np
import pandas as pd

from sensoryrec.recommenders.base import rank
from sensoryrec.refinement import shift
from sensoryrec.text import content_words


def refinement_buttons(lexicons):
    """(label, trait, sign) for every refinement, in the order of the lexicon
    file: e.g. ("sweeter", "sweetness", +1), ("drier", "sweetness", -1)."""
    buttons = []
    for trait, lex in lexicons.items():
        buttons += [(lex["more"], trait, 1), (lex["less"], trait, -1)]
    return buttons


def chosen_alpha(path, default):
    """The step size chosen on the dev queries by `make refine`, or `default`
    (marked as such on screen) while that file does not exist."""
    try:
        table = pd.read_csv(path)
    except FileNotFoundError:
        return default, False
    return float(table.loc[table["chosen"], "alpha"].iloc[0]), True


def cheaper_limit(top):
    """'Cheaper' = at most the median price of the wines on screen (as in
    `make refine`). None if none of them has a price."""
    prices = top["price"].dropna()
    return float(prices.median()) if len(prices) else None


# what is left of "isn't", "doesn't", ... once the apostrophe is removed
CONTRACTION_STUBS = {"isn", "aren", "wasn", "weren", "doesn", "don", "didn", "won", "wouldn", "shouldn", "couldn"}


def shared_words(query, note):
    """Content words the request and the tasting note have in common, sorted:
    a first, honest hint of why a wine was picked (not an explanation)."""
    return sorted((content_words(query) & content_words(note)) - CONTRACTION_STUBS)


def recommend(model, query, k, filters, steps=(), refiner=None, alpha=None):
    """Top k of `model` for `query`, after the taste refinement steps.

    steps: (trait, sign) pairs, applied in order to the query vector. They
    need the Semantic model and its Refiner; for other models steps must be
    empty (the screen only offers them for the Semantic model)."""
    if not steps:
        return model.recommend(query, k=k, filters=filters)
    if refiner is None or alpha is None:
        raise ValueError("taste refinement needs the Semantic model's Refiner and alpha")
    q = model.encode([model.query_prefix + query])[0]
    for trait, sign in steps:
        q = shift(q, refiner.directions[trait], alpha, sign)
    return rank(model.catalogue, refiner.scores(q), k, filters)
