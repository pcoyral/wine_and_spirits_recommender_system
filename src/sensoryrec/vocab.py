"""Checks every casual query before it is used (CLAUDE.md rule 5).

1. Vocabulary check: no "expert" term from data/wine_vocab.txt.
2. Leak check: no name of the query's own wine (variety, winery, region,
   province, country, designation), so "no name, region or variety" is
   verified, not only instructed (DECISIONS.md).
"""
from pathlib import Path

import pandas as pd

from sensoryrec.text import find_terms, normalise

LEAK_COLUMNS = ["variety", "winery", "region_1", "region_2", "province", "country", "designation"]

# Names that are also everyday words; a query may use them without leaking
# ("us" is a pronoun; a rosé may be called "rosé" or "pink", D7).
NOT_A_LEAK = {"us", "other", "unknown"}


def load_vocab(path):
    """{"expert": [...], "everyday": [...]} from the [expert] / [everyday]
    sections of the file. Lines starting with # are comments."""
    tiers, current = {"expert": [], "everyday": []}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            continue
        tiers[current].append(line)
    return tiers


def expert_terms_used(query, vocab):
    """Banned expert terms found in the query."""
    return find_terms(query, vocab["expert"])


def leaked_names(query, wine, everyday=()):
    """Names of the query's own wine that appear in the query. `wine` is a
    row (dict or Series) of the catalogue. Names that are everyday words
    (e.g. the variety "Rosé") are not counted as leaks."""
    allowed = NOT_A_LEAK | {normalise(w) for w in everyday}
    names = []
    for col in LEAK_COLUMNS:
        value = wine.get(col)
        if value is None or pd.isna(value) or normalise(value) in allowed:
            continue
        names.append(str(value))
    return find_terms(query, names)


def violations(query, wine, vocab):
    """Every problem with a query: banned terms plus leaked names."""
    return expert_terms_used(query, vocab) + leaked_names(query, wine, vocab["everyday"])
