"""Small text helpers shared by the vocabulary check, the query generator,
the Attribute recommender and refinement.

All matching works on normalised text (lower case, no accents, only letters,
digits and single spaces), so "Rosé", "rose" and "ROSÉ" are the same word,
and "full-bodied" becomes "full bodied".
"""
import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# A term right after one of these words does not count ("not sweet").
NEGATORS = ["not", "no", "without", "never"]


def normalise(text):
    """Lower case, no accents, only letters, digits and single spaces:
    "Nero d'Avola" -> "nero d avola", "Rosé" -> "rose"."""
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def term_pattern(term, negation_guard=False):
    """Regex for a (possibly multi-word) term as whole words, plural allowed
    ("tannin" also matches "tannins"). With negation_guard, a match right
    after a negator ("not", "no", ...) is ignored."""
    guard = "".join(f"(?<!\\b{n} )" for n in NEGATORS) if negation_guard else ""
    return re.compile(guard + r"\b" + re.escape(normalise(term)) + r"(?:s|es)?\b")


def any_term_pattern(terms, negation_guard=False):
    """One regex matching any of the terms (whole words, plural allowed);
    faster than one regex per term when scanning many notes."""
    guard = "".join(f"(?<!\\b{n} )" for n in NEGATORS) if negation_guard else ""
    options = "|".join(re.escape(normalise(t)) for t in sorted(terms, key=len, reverse=True))
    return re.compile(guard + r"\b(?:" + options + r")(?:s|es)?\b")


def find_terms(text, terms, negation_guard=False):
    """The terms that occur in `text`, in the order of `terms`."""
    text = normalise(text)
    return [t for t in terms if term_pattern(t, negation_guard).search(text)]


# In almost every query and note, so they say nothing about copying.
GENERIC_WORDS = {"wine", "wines"}


def content_words(text):
    """Words of 3+ letters that are not English stop words or "wine"."""
    return {w for w in normalise(text).split()
            if len(w) >= 3 and w not in ENGLISH_STOP_WORDS and w not in GENERIC_WORDS}


def overlap_share(query, note):
    """Share of the query's content words that also appear in the note:
    how much the query copies its tasting note (0 = nothing, 1 = everything)."""
    words = content_words(query)
    if not words:
        return 0.0
    return len(words & content_words(note)) / len(words)
