"""`make worked-example`: the demo's one complete worked example, end to end
from a script (CLAUDE.md, REQUIRED; S3).

persona -> casual request -> top 5 with bottle pictures -> one refinement
("sweeter", say) -> new top 5 -> grounded explanation of each turn.
Persona, request, budget and refinement are in config.yaml (rag.worked_example).

Outputs
- results/worked_example.md    readable, for the thesis
- results/worked_example.html  the same with the bottle pictures, for the demo
- results/worked_example.csv   one row per wine and turn
"""
import base64
import html
import logging

import pandas as pd

from sensoryrec import images, prototype, rag
from sensoryrec.config import load_config, repo_path

log = logging.getLogger(__name__)


def picture_html(row, colour, shapes, photos, height=150):
    """The real photo where we have one (logged in data/images/sources.csv), else the drawn bottle."""
    if row.wine_id in photos:
        data = base64.b64encode(repo_path(photos[row.wine_id]).read_bytes()).decode()
        return f'<img src="data:image/jpeg;base64,{data}" alt="bottle photo" style="height:{height}px">'
    return images.bottle_svg(images.bottle_shape(row.variety, colour, shapes), colour, height)


def photo_files(sources_path):
    """wine_id -> photo file; a photo the student approved comes first."""
    if not sources_path.exists():
        return {}
    t = pd.read_csv(sources_path)
    t = t.assign(order=(~t["source"].astype(str).str.startswith("Open Food Facts")).astype(int))
    t = t.sort_values("order").drop_duplicates("wine_id")
    return dict(zip(t["wine_id"], t["file"]))


def run(model, refiner, alpha, llm, template, ex, k):
    """The two turns. Returns a list of (turn label, top, explanation)."""
    filters = {"max_price": ex["max_price"]} if ex.get("max_price") else {}
    first = prototype.recommend(model, ex["request"], k, filters)
    turns = [("1. First answer", first, rag.explain(llm, template, ex["request"], first))]
    trait, sign = ex["refinement"]["trait"], ex["refinement"]["sign"]
    second = prototype.recommend(model, ex["request"], k, filters, [(trait, sign)], refiner, alpha)
    change = rag.change_text(ex["refinement"]["label"], first["wine_id"].tolist(), second["wine_id"].tolist())
    turns.append((f'2. After "{ex["refinement"]["label"]}"', second,
                  rag.explain(llm, template, ex["request"], second, change)))
    return turns


def to_markdown(ex, turns, engine):
    lines = ["# Worked example", "", f"**Persona.** {ex['persona']}", "",
             f"**Request.** “{ex['request']}”" + (f" (budget: at most ${ex['max_price']})" if ex.get("max_price") else ""),
             "", f"Engine: {engine}. Ranking is deterministic; the LLM only writes the explanations.", ""]
    for label, top, out in turns:
        lines += [f"## {label}", ""]
        if out["intro"]:
            lines += [out["intro"], ""]
        if out["source"] != "llm":
            lines += ["*Explanation fallback (no LLM answer): words shared with the request.*", ""]
        for r, why in zip(top.itertuples(), out["whys"]):
            price = f"${r.price:.0f}" if pd.notna(r.price) else "price unknown"
            lines += [f"{r.rank}. **{r.title}** ({r.variety}, {r.country}, {price}, {r.points} pts)",
                      f"   - Tasting note: {r.description}", f"   - Why: {why}", ""]
    return "\n".join(lines)


def to_html(ex, turns, engine, colours, shapes, photos):
    e = html.escape
    parts = [f"<h1>Worked example</h1><p><b>Persona.</b> {e(ex['persona'])}</p>",
             f"<p><b>Request.</b> “{e(ex['request'])}”</p><p class=meta>Engine: {e(engine)}. "
             "Ranking is deterministic; the LLM only writes the explanations.</p>"]
    for label, top, out in turns:
        parts.append(f"<h2>{e(label)}</h2>")
        if out["intro"]:
            parts.append(f"<p class=intro>{e(out['intro'])}</p>")
        for r, why in zip(top.itertuples(), out["whys"]):
            price = f"${r.price:.0f}" if pd.notna(r.price) else "price unknown"
            parts.append(
                f'<div class=card><div class=pic>{picture_html(r, colours.get(r.variety), shapes, photos)}</div>'
                f'<div><div class=name>{r.rank}. {e(r.title)}</div>'
                f'<div class=meta>{e(str(r.variety))} · {e(str(r.country))} · {price} · {r.points} pts</div>'
                f'<div class=note>{e(r.description)}</div><div class=why><b>Why:</b> {e(why)}</div></div></div>')
    style = ("body{font-family:Inter,Arial,sans-serif;background:#F8F4EB;color:#292923;max-width:860px;"
             "margin:auto;padding:16px}h1,h2{font-family:'Cormorant Garamond',Georgia,serif;color:#49272D}"
             ".card{display:grid;grid-template-columns:90px 1fr;gap:16px;background:#fff;padding:14px;margin:10px 0;"
             "border-left:3px solid #B59A68}.pic{display:flex;align-items:flex-end;justify-content:center}"
             ".name{font-family:Georgia,serif;font-size:1.1rem}.meta{font-size:.8rem;color:#6b665e}"
             ".note{margin:6px 0;font-size:.9rem}.why{font-size:.9rem;background:#F8F4EB;padding:6px}")
    return f"<!doctype html><meta charset=utf-8><title>Worked example</title><style>{style}</style>{''.join(parts)}"


def main():
    from sensoryrec.recommenders import build_models
    from sensoryrec.refinement import Refiner, load_lexicons
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    c = cfg["rag"]
    ex = c["worked_example"]
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet")
    test_ids = pd.read_csv(processed / "test_ids.csv")["wine_id"]
    model = next(m for m in build_models(cfg) if m.name == c["engine"]).fit(wines, holdout_ids=test_ids)
    lexicons = load_lexicons(repo_path(cfg["refinement"]["lexicons"]))
    refiner = Refiner(model.embeddings, model.catalogue, lexicons).fit(test_ids)
    alpha, from_dev = prototype.chosen_alpha(repo_path(cfg["refinement"]["out_dir"]) / "refinement_alpha_dev.csv",
                                             default=min(cfg["refinement"]["alphas"]))
    if not from_dev:
        log.warning("refinement step size not chosen on dev yet: using %s", alpha)

    turns = run(model, refiner, alpha, rag.from_config(cfg), c["prompt"], ex, c["k"])
    colour_table = pd.read_csv(repo_path(cfg["attribute"]["colour_table"]))
    colours = dict(zip(colour_table["variety"], colour_table["colour"]))
    shapes = images.load_shapes(repo_path(cfg["images"]["shapes"]))
    photos = photo_files(repo_path(cfg["images"]["sources"]))

    out = repo_path(cfg["eval"]["out_dir"])
    (out / "worked_example.md").write_text(to_markdown(ex, turns, c["engine"]), encoding="utf-8")
    (out / "worked_example.html").write_text(to_html(ex, turns, c["engine"], colours, shapes, photos), encoding="utf-8")
    rows = [{"turn": label, "rank": r.rank, "wine_id": r.wine_id, "title": r.title, "why": why,
             "source": o["source"], "has_photo": r.wine_id in photos}
            for label, top, o in turns for r, why in zip(top.itertuples(), o["whys"])]
    pd.DataFrame(rows).to_csv(out / "worked_example.csv", index=False)
    log.info("worked example written to %s (worked_example.md/.html/.csv)", out)
    if any(o["source"] != "llm" for _, _, o in turns):
        raise SystemExit("An explanation fell back (no LLM answer): run again later; the pages show the fallback.")


if __name__ == "__main__":
    main()
