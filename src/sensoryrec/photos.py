"""Real bottle photos for a curated demo set, checked by the student
(increment A; DECISIONS.md, 16 Nov 2026).

`make photos` (needs internet; runs in the Photos workflow)
  For every wine in the demo set (the wines the demo shows, plus any
  wine_id listed in data/images/demo_extra_ids.csv) it searches Open Food
  Facts, an open product database whose photos may be reused with credit
  (CC BY-SA), and keeps the candidate whose name shares the most words with
  our title. It saves a small copy of each candidate photo and a review
  sheet, data/images/photo_review.csv, with an empty `approved` column.
  Nothing is used yet.

`make photos-apply`
  Uses only the rows the student marked `approved` = yes: the photo goes to
  data/images/photos/ and its credit (source page, licence) to
  data/images/sources.csv. The app shows, in this order: an approved photo,
  an X-Wines label photo (CC0), the drawn bottle.

No shop or site whose terms forbid it is used (CLAUDE.md).
"""
import json
import logging
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from sensoryrec.config import load_config, repo_path
from sensoryrec.images import GENERIC_NAME_WORDS, normalise, thumbnail, update_sources

log = logging.getLogger(__name__)
SEARCH = "https://search.openfoodfacts.org/search"          # the newer search service
PRODUCT_API = "https://world.openfoodfacts.org/api/v2/product/{code}.json?fields=code,product_name,brands,image_front_url"
PRODUCT_PAGE = "https://world.openfoodfacts.org/product/{code}"
LICENCE = "CC BY-SA 3.0 (Open Food Facts contributors)"
# A generic name: no personal details are sent to the external service.
USER_AGENT = "SensoryRec-research-prototype/0.1"


def get_json(url, waits=(10, 30, 60)):
    """GET a JSON answer; on "busy" (429/503) wait and try again."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for wait in [*waits, None]:
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in (429, 502, 503) or wait is None:
                raise
            log.info("Open Food Facts busy (HTTP %d), waiting %d s", error.code, wait)
            time.sleep(wait)


def download(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)


def search_terms(wine):
    """Winery plus the distinctive words of the title (no year, no region in
    brackets)."""
    title = str(wine["title"]).split("(")[0]
    words = [w for w in normalise(title) if not w.isdigit()]
    return " ".join(dict.fromkeys(words))


def overlap(title, candidate_name, brands):
    """Share of our title's distinctive words found in the candidate's name
    and brand (0-1). Used only to choose which candidate to show for review."""
    ours = {w for w in normalise(str(title).split("(")[0]) if not w.isdigit()} - GENERIC_NAME_WORDS
    theirs = set(normalise(f"{candidate_name} {brands}"))
    return len(ours & theirs) / len(ours) if ours else 0.0


def brand_text(brands):
    return " ".join(brands) if isinstance(brands, list) else str(brands or "")


def best_candidate(wine, products):
    """The product whose name and brand overlap our title most, or None.
    (Search hits may lack the photo address; it is fetched afterwards.)"""
    scored = [(overlap(wine["title"], p.get("product_name", ""), brand_text(p.get("brands"))), p)
              for p in products if p.get("code")]
    scored = [x for x in scored if x[0] > 0]
    if not scored:
        return None, 0.0
    score, product = max(scored, key=lambda x: x[0])
    return product, score


def demo_set(cfg):
    """wine_ids the app's engines show in the demo, plus the student's extra list."""
    demo = pd.read_csv(repo_path(cfg["demo"]["out_dir"]) / f"demo_top{cfg['k']}.csv")
    models = cfg["photos"].get("models")
    if models and "model" in demo:
        demo = demo[demo["model"].isin(models)]     # the engines the app shows
    ids = set(demo["wine_id"])
    extra = repo_path(cfg["photos"]["extra_ids"])
    if extra.exists():
        ids |= set(pd.read_csv(extra)["wine_id"])
    return sorted(ids)


def find(cfg, get=get_json, fetch=download):
    """Search a candidate photo for every demo wine; write the review sheet."""
    c = cfg["photos"]
    wines = pd.read_parquet(repo_path(cfg["paths"]["processed_dir"]) / "wines.parquet").set_index("wine_id")
    cand_dir = repo_path(c["candidates_dir"])
    cand_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for wine_id in demo_set(cfg):
        wine = wines.loc[wine_id]
        params = {"q": f'{search_terms(wine)} categories_tags:"en:wines"', "page_size": 20,
                  "fields": "code,product_name,brands"}
        try:
            products = get(f"{SEARCH}?{urllib.parse.urlencode(params)}").get("hits", [])
        except Exception as error:                 # one failed search must not stop the others
            log.warning("search failed for %s: %s", wine_id, error)
            products = []
        product, score = best_candidate(wine, products)
        if product is not None:
            try:
                product = get(PRODUCT_API.format(code=product["code"])).get("product") or {}
                product.setdefault("code", product.get("code"))
            except Exception as error:
                log.warning("product lookup failed for %s: %s", wine_id, error)
                product = None
            if product is not None and not product.get("image_front_url"):
                product = None                     # no photo on that product
        row = {"wine_id": wine_id, "our_title": wine["title"], "candidate_name": None, "candidate_brand": None,
               "candidate_page": None, "image_url": None, "word_overlap": round(score, 2), "photo": None,
               "approved": ""}
        if product is not None:
            path = cand_dir / f"{wine_id}.jpg"
            try:
                raw = cand_dir / f"{wine_id}.download"
                fetch(product["image_front_url"], raw)
                thumbnail(raw, path)
                raw.unlink()
                row.update(candidate_name=product.get("product_name"),
                           candidate_brand=brand_text(product.get("brands")),
                           candidate_page=PRODUCT_PAGE.format(code=product["code"]),
                           image_url=product["image_front_url"], photo=str(path.relative_to(repo_path("."))))
            except Exception as error:
                log.warning("image download failed for %s: %s", wine_id, error)
        rows.append(row)
        time.sleep(c["pause_seconds"])             # Open Food Facts asks for at most ~10 searches a minute
    sheet = pd.DataFrame(rows)
    sheet.to_csv(repo_path(c["review"]), index=False)
    log.info("%d demo wines, %d with a candidate photo", len(sheet), int(sheet["photo"].notna().sum()))
    return sheet


def apply(cfg):
    """Use the approved photos only; credit each in data/images/sources.csv."""
    c = cfg["photos"]
    sheet = pd.read_csv(repo_path(c["review"]), dtype={"approved": str})
    approved = sheet[sheet["approved"].astype(str).str.strip().str.lower().isin({"yes", "y", "1", "true"})]
    out = repo_path(c["photos_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in approved.itertuples():
        target = out / f"{r.wine_id}.jpg"
        shutil.copyfile(repo_path(r.photo), target)
        rows.append({"wine_id": r.wine_id, "file": str(target.relative_to(repo_path("."))),
                     "source": f"Open Food Facts: {r.candidate_name} ({r.candidate_brand})",
                     "url": r.candidate_page, "licence": LICENCE,
                     "matched_by": "checked by eye by the student (data/images/photo_review.csv)"})
    update_sources(repo_path(cfg["images"]["sources"]), rows, "Open Food Facts")
    log.info("%d approved photos applied", len(rows))
    return rows


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Real bottle photos for the demo wines")
    parser.add_argument("--apply", action="store_true", help="use the photos approved in the review sheet")
    cfg = load_config()
    if parser.parse_args().apply:
        apply(cfg)
    else:
        find(cfg)


if __name__ == "__main__":
    main()
