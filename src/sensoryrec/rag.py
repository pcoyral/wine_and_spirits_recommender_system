"""Grounded explanations for the recommended wines (S3; plan approved by the
student, 19 Nov 2026, DECISIONS.md).

Retrieval-augmented generation, in its simplest form:
1. retrieval: the engine (Sentence-BERT by default, `rag.engine`) ranks the
   wines; the LLM never chooses, adds, drops or reorders them (CLAUDE.md
   rule 2);
2. augmentation: the prompt holds the request and, for each of the k wines,
   its title, grape, country, price and Wine Enthusiast tasting note, and
   nothing else;
3. generation: Gemini Flash-Lite writes one intro sentence and one sentence
   per wine, each quoting a phrase of that wine's note, as JSON. Temperature
   0, model pinned in config.yaml, every answer cached in data/llm_cache/.

If the LLM cannot answer (no key, quota used up, unreadable answer) the
explanation falls back to the words the note shares with the request,
labelled as such, so the app always shows something honest.

`make faithfulness`: 50 simulated requests -> top 5 -> explanation, written
to results/faithfulness_sample.csv for the student to annotate (faithful
yes / partly / no; claims not in the notes). An automatic column counts how
many quoted phrases really occur in the notes.
`make worked-example`: the demo's complete worked example, see worked_example.py.
"""
import json
import logging
import re

import pandas as pd

from sensoryrec import llm as llm_module
from sensoryrec.prototype import shared_words

log = logging.getLogger(__name__)


def describe(top):
    """The k retrieved wines, numbered in rank order, as the LLM sees them."""
    lines = []
    for i, r in enumerate(top.itertuples(), start=1):
        price = f"${r.price:.0f}" if pd.notna(r.price) else "price unknown"
        lines.append(f"Wine {i}: {r.title}. {r.variety}, {r.country}, {price}. "
                     f"Tasting note: \"{r.description}\"")
    return "\n".join(lines)


def change_text(label, before_ids, after_ids):
    """What the follow-up asked for and which wines are new, for the refinement
    turn's prompt ("" on the first turn)."""
    if not label:
        return ""
    new = [i for i, w in enumerate(after_ids, start=1) if w not in set(before_ids)]
    which = ", ".join(f"Wine {i}" for i in new) if new else "none (same wines, new order)"
    return (f'They then asked for the selection to be "{label}". The list below is the new selection. '
            f"Wines that were not in the previous selection: {which}. "
            "In the intro, say in a few words what changed.")


def build_prompt(template, request, top, change=""):
    return template.format(request=request, wines=describe(top), k=len(top), change=change)


def parse(text, k):
    """The LLM's JSON answer -> {"intro": str, "whys": [k strings]}; raises
    ValueError if anything is missing, so a broken answer is never shown."""
    data = json.loads(text)
    intro = str(data["intro"]).strip()
    by_number = {int(w["wine"]): str(w["why"]).strip() for w in data["wines"]}
    if sorted(by_number) != list(range(1, k + 1)) or not intro or not all(by_number.values()):
        raise ValueError(f"expected an intro and wines 1..{k}, got {sorted(by_number)}")
    return {"intro": intro, "whys": [by_number[i] for i in range(1, k + 1)]}


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def quote_check(why, note):
    """(quotes in the sentence, of which found word for word in the note).
    Only double-quoted phrases of 2+ words count; punctuation and case are
    ignored. A first, automatic faithfulness signal, not a replacement for
    reading the explanation."""
    quotes = [q for q in re.findall(r'["“”]([^"“”]+)["“”]', str(why)) if len(_norm(q).split()) >= 2]
    found = sum(_norm(q) in _norm(note) for q in quotes)
    return len(quotes), found


def fallback(request, top, reason):
    """The honest explanation when the LLM cannot answer: shared words."""
    whys = []
    for r in top.itertuples():
        words = shared_words(request, r.description)
        whys.append("Words your request shares with the tasting note: " + ", ".join(words) + "."
                    if words else "No shared words: matched on the meaning of the note.")
    return {"intro": "", "whys": whys, "source": "fallback", "reason": reason}


def explain(llm, template, request, top, change=""):
    """Explanation for the wines in `top` (already ranked). Never raises:
    any LLM problem gives the labelled fallback."""
    if top.empty:
        return {"intro": "", "whys": [], "source": "fallback", "reason": "no wines"}
    prompt = build_prompt(template, request, top, change)
    try:
        out = parse(llm.generate(prompt), len(top))
    except (Exception, SystemExit) as error:      # no key, quota, network, bad JSON
        log.warning("explanation fell back to shared words: %s", str(error)[:200])
        return fallback(request, top, str(error)[:200])
    return {**out, "source": "llm", "reason": ""}


def from_config(cfg, call=None):
    """The cached Gemini LLM that writes explanations (section rag.llm)."""
    return llm_module.gemini_from_config(cfg, call=call, section="rag")


def faithfulness_rows(requests, model, llm, template, k, filters=None):
    """One row per request: the request, the wines with their notes, the
    explanation, the automatic quote check, and empty columns to annotate."""
    rows = []
    for r in requests.itertuples():
        top = model.recommend(r.query, k=k, filters=filters)
        out = explain(llm, template, r.query, top)
        n_quotes = n_found = 0
        for why, note in zip(out["whys"], top["description"]):
            q, f = quote_check(why, note)
            n_quotes, n_found = n_quotes + q, n_found + f
        rows.append({
            "case": len(rows) + 1, "query_id": r.query_id, "request": r.query,
            "wines": describe(top),
            "explanation": "\n".join([out["intro"], *[f"Wine {i}: {w}" for i, w in enumerate(out["whys"], 1)]]).strip(),
            "source": out["source"], "quotes": n_quotes, "quotes_found_in_notes": n_found,
            "faithful_yes_partly_no": "", "claims_not_in_notes": "",
        })
    return pd.DataFrame(rows)


def main():
    """`make faithfulness` -> results/faithfulness_sample.csv"""
    from sensoryrec.config import load_config, repo_path
    from sensoryrec.recommenders import build_models
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    c = cfg["rag"]
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")["wine_id"]
    model = next(m for m in build_models(cfg) if m.name == c["engine"]).fit(wines, holdout_ids=test_ids)

    queries = pd.read_csv(repo_path(cfg["simulation"]["out"]))
    usable = queries[~queries["flagged"].astype(bool)]
    requests = usable.sample(n=c["n_faithfulness"], random_state=cfg["seed"]).sort_values("query_id")
    llm = from_config(cfg)
    sheet = faithfulness_rows(requests, model, llm, c["prompt"], c["k"])
    out = repo_path(c["faithfulness_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False)
    n_llm = int((sheet["source"] == "llm").sum())
    log.info("%d explanations (%d from the LLM, %d fallback); quotes found in notes: %d / %d -> %s",
             len(sheet), n_llm, len(sheet) - n_llm, sheet["quotes_found_in_notes"].sum(), sheet["quotes"].sum(), out)
    if n_llm < len(sheet):
        raise SystemExit("Some explanations fell back (quota or answer problem): run again later; "
                         "answers so far are in the cache.")


if __name__ == "__main__":
    main()
