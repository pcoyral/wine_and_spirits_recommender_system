"""Increment A: what a wine looks like in the app (DECISIONS.md, 16 Nov 2026).

1. A bottle **shape** for every wine: the classic shapes carry information a
   novice can learn (a Bordeaux bottle has high shoulders, a Burgundy bottle
   sloping ones, Riesling comes in a tall flute). Shape from
   data/bottle_shapes.csv (by grape, drafted by Claude, to be checked by the
   student), else from the wine's colour; drawn as an SVG tinted by colour.
   No photo, so no licence question.
2. A real **label image** where one exists: X-Wines (CC0) publishes label
   photos for its 1,007-wine Slim version. `match_xwines` links a Wine
   Enthusiast wine to an X-Wines wine only when the winery, a word of the
   wine's name or grape, and the vintage all agree. Every image used is
   listed in data/images/sources.csv with its source and licence (CLAUDE.md).
"""
import ast
import re
import unicodedata

import pandas as pd

COLOUR_HEX = {"red": "#5a2a31", "white": "#cbb67c", "rosé": "#d98b98", "sparkling": "#bfa56a", None: "#8a7a5a"}
DEFAULT_SHAPE = {"red": "bordeaux", "white": "burgundy", "rosé": "flute", "sparkling": "champagne"}

# Outline of each shape on a 40 x 140 canvas: (neck width, shoulder start, shoulder end, body width)
SHAPES = {
    "bordeaux":  (9, 34, 46, 30),   # straight neck, high square shoulders
    "burgundy":  (9, 30, 70, 32),   # long sloping shoulders
    "flute":     (8, 30, 80, 24),   # tall and slim
    "champagne": (11, 34, 70, 34),  # wide, heavy glass, sloping
    "port":      (10, 40, 52, 30),  # short neck, bulb under the shoulder
}


def bottle_shape(variety, colour, table):
    """Shape name for a wine: by grape if listed, else by colour, else Bordeaux."""
    shape = table.get(variety)
    if shape in SHAPES:
        return shape
    return DEFAULT_SHAPE.get(colour, "bordeaux")


def bottle_svg(shape, colour, height=140):
    """An SVG bottle of the given shape, tinted by wine colour."""
    neck, s0, s1, body = SHAPES[shape]
    cx, top, bottom = 20, 4, 136
    fill = COLOUR_HEX.get(colour, COLOUR_HEX[None])
    path = (f"M{cx - neck / 2},{top} L{cx + neck / 2},{top} L{cx + neck / 2},{s0} "
            f"C{cx + neck / 2},{(s0 + s1) / 2} {cx + body / 2},{(s0 + s1) / 2} {cx + body / 2},{s1} "
            f"L{cx + body / 2},{bottom - 3} Q{cx + body / 2},{bottom} {cx + body / 2 - 3},{bottom} "
            f"L{cx - body / 2 + 3},{bottom} Q{cx - body / 2},{bottom} {cx - body / 2},{bottom - 3} "
            f"L{cx - body / 2},{s1} C{cx - body / 2},{(s0 + s1) / 2} {cx - neck / 2},{(s0 + s1) / 2} "
            f"{cx - neck / 2},{s0} Z")
    label_top = max(s1 + 8, 80)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 140" height="{height}" '
            f'role="img" aria-label="{shape} bottle">'
            f'<path d="{path}" fill="{fill}"/>'
            f'<rect x="{cx - neck / 2 - 0.5}" y="{top}" width="{neck + 1}" height="12" fill="#B59A68"/>'
            f'<rect x="{cx - body / 2 + 3}" y="{label_top}" width="{body - 6}" height="26" fill="#FFFDF7"/>'
            f'<rect x="{cx - body / 2 + 3}" y="{label_top}" width="{body - 6}" height="2" fill="#B59A68"/></svg>')


def normalise(text):
    """Lower case, no accents, letters and digits only."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def vintage(title):
    """The first year 1900-2029 in a Wine Enthusiast title, or None."""
    m = re.search(r"\b(19\d\d|20[0-2]\d)\b", str(title))
    return int(m.group(1)) if m else None


def match_xwines(ours, theirs):
    """Wine Enthusiast wine_id -> X-Wines WineID, only for confident matches:
    same winery (normalised), at least one word of the X-Wines wine name or a
    grape present in our title or variety, and our vintage among the X-Wines
    vintages. ours: wine_id, winery, title, variety; theirs: WineID,
    WineryName, WineName, Grapes, Vintages."""
    by_winery = {}
    for r in theirs.itertuples():
        by_winery.setdefault(" ".join(normalise(r.WineryName)), []).append(r)
    matches = {}
    for w in ours.itertuples():
        candidates = by_winery.get(" ".join(normalise(w.winery)), [])
        if not candidates:
            continue
        words = set(normalise(w.title)) | set(normalise(w.variety))
        year = vintage(w.title)
        found = []
        for c in candidates:
            name_words = set(normalise(c.WineName)) - {"red", "white", "wine", "reserva", "reserve"}
            grapes = {g for grape in ast.literal_eval(c.Grapes) for g in normalise(grape)}
            years = set(ast.literal_eval(c.Vintages)) if isinstance(c.Vintages, str) else set()
            if (name_words & words or grapes & words) and year in years:
                found.append(c.WineID)
        if len(found) == 1:            # ambiguous matches are skipped
            matches[w.wine_id] = found[0]
    return matches


def load_shapes(path):
    t = pd.read_csv(path)
    return dict(zip(t["variety"], t["shape"]))


def main():
    """`make images`: copy X-Wines label photos (CC0) for the wines that
    match, log their source and licence, and report the coverage.
    Needs data/raw/xwines_slim/ with XWines_Slim_1K_wines.csv and the
    unzipped label images (named <WineID>.<ext>)."""
    import logging
    import shutil

    from sensoryrec.config import load_config, repo_path
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)
    cfg = load_config()
    c = cfg["images"]
    src = repo_path(c["xwines_slim_dir"])
    theirs = pd.read_csv(next(src.glob("*Slim_1K_wines.csv")))
    labels = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    ours = pd.read_parquet(repo_path(cfg["paths"]["processed_dir"]) / "wines.parquet")
    matches = {w: x for w, x in match_xwines(ours, theirs).items() if str(x) in labels}

    out = repo_path(c["labels_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for wine_id, xw_id in sorted(matches.items()):
        image = labels[str(xw_id)]
        target = out / f"{wine_id}{image.suffix.lower()}"
        shutil.copyfile(image, target)
        rows.append({"wine_id": wine_id, "file": str(target.relative_to(repo_path("."))),
                     "source": f"X-Wines Slim label image, WineID {xw_id}",
                     "url": "https://github.com/rogerioxavier/X-Wines", "licence": "CC0 1.0",
                     "matched_by": "winery + name/grape word + vintage (sensoryrec.images.match_xwines)"})
    pd.DataFrame(rows, columns=["wine_id", "file", "source", "url", "licence", "matched_by"]) \
        .to_csv(repo_path(c["sources"]), index=False)
    coverage = pd.DataFrame([("wines_in_catalogue", len(ours)), ("xwines_slim_wines", len(theirs)),
                             ("label_images_available", len(labels)), ("wines_with_label_image", len(rows)),
                             ("wines_with_drawn_bottle", len(ours))], columns=["metric", "value"])
    coverage.to_csv(repo_path(c["coverage"]), index=False)
    log.info("\n%s", coverage.to_string(index=False))


if __name__ == "__main__":
    main()
