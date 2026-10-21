"""Checks every casual query before it is used (CLAUDE.md rule 5).

1. Vocabulary check: no "expert" term from data/wine_vocab.txt.
2. Leak check: no name of the query's own wine, so "no name, region or
   variety" is verified, not only instructed (DECISIONS.md):
   - the full variety, winery, region, province, country and designation;
   - for variety, region and province also their distinctive single words
     ("cabernet" from "Cabernet Sauvignon", "willamette" from "Willamette
     Valley"). Not for winery and designation, whose single words are often
     ordinary ("Black Stallion" would flag "black cherry").
The full set of checks applied to a query is queries.problems().
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sensoryrec.text import find_terms, normalise

LEAK_COLUMNS = ["variety", "winery", "region_1", "region_2", "province", "country", "designation"]
PARTIAL_COLUMNS = ["variety", "region_1", "region_2", "province"]

# Names that are also everyday words; a query may use them without leaking
# ("us" is a pronoun, "other" a placeholder in the data).
NOT_A_LEAK = {"us", "other", "unknown"}

# Words inside variety and place names that say nothing about the wine's
# identity, so they are not leaks on their own.
GENERIC_NAME_WORDS = {
    "valley", "county", "hills", "hill", "coast", "coastal", "river", "mountain", "mountains",
    "lake", "lakes", "north", "south", "east", "west", "northern", "southern", "eastern", "western",
    "central", "upper", "lower", "foothills", "highlands", "ridge", "creek", "island", "islands",
    "bench", "mesa", "mount", "mont", "monte", "village", "villages", "saint", "sainte", "santa",
    "blanc", "noir", "rouge", "rosso", "bianco", "blanco", "tinto", "vino", "style", "blend",
    "grand", "superiore", "classico", "riserva", "reserva",
    # ordinary English words found in the test/dev wines' names
    "great", "fair", "play", "province", "region", "district", "peninsula", "southwest",
    "northeast", "northeastern", "southeast", "northwest", "vale", "cape", "mission",
    "petit", "petite", "grave",
}


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


def is_ordinary(word, everyday):
    return word in everyday or word in ENGLISH_STOP_WORDS or word in NOT_A_LEAK


def usable_name(name, everyday):
    """A name can be checked if, normalised, it has 3+ characters and is not
    made only of everyday words or stop words ("[+]" -> "", "Dry White")."""
    words = normalise(name).split()
    return len(" ".join(words)) >= 3 and not all(is_ordinary(w, everyday) for w in words)


def distinctive_words(name, everyday):
    """Words of a variety or place name that identify it: 4+ letters, not
    ordinary, not generic ("cabernet", "willamette"; not "valley")."""
    return [w for w in normalise(name).split()
            if len(w) >= 4 and not is_ordinary(w, everyday) and w not in GENERIC_NAME_WORDS]


def leaked_names(query, wine, everyday=()):
    """Names (or distinctive words of names) of the query's own wine that
    appear in the query. `wine` is a row (dict or Series) of the catalogue."""
    everyday = {normalise(w) for w in everyday}
    names, words = [], []
    for col in LEAK_COLUMNS:
        value = wine.get(col)
        if value is None or pd.isna(value) or not usable_name(value, everyday):
            continue
        names.append(str(value))
        if col in PARTIAL_COLUMNS:
            words += [w for w in distinctive_words(value, everyday) if w not in words]
    found = find_terms(query, names)
    covered = {w for name in found for w in normalise(name).split()}
    return found + [w for w in find_terms(query, words) if w not in covered]
