import pandas as pd
import pytest
from PIL import Image

from sensoryrec import photos
from sensoryrec.images import update_sources

WINE = pd.Series({"title": "Georges Duboeuf 2013 Flower Label  (Beaujolais)", "winery": "Georges Duboeuf"})


def test_search_terms_drop_year_and_region():
    assert photos.search_terms(WINE) == "georges duboeuf flower label"


def test_best_candidate_needs_an_image_and_shared_words():
    products = [{"code": "1", "product_name": "Beaujolais Villages", "brands": "Georges Duboeuf", "image_front_url": "u1"},
                {"code": "2", "product_name": "Flower Label", "brands": "Georges Duboeuf", "image_front_url": "u2"},
                {"code": "3", "product_name": "Flower Label", "brands": "Georges Duboeuf"},          # no image
                {"code": "4", "product_name": "Cola", "brands": "Other", "image_front_url": "u4"}]
    product, score = photos.best_candidate(WINE, products)
    assert product["code"] == "2" and score == 1.0
    assert photos.best_candidate(WINE, [products[3]]) == (None, 0.0)


def test_update_sources_keeps_other_sources(tmp_path):
    path = tmp_path / "sources.csv"
    update_sources(path, [{"wine_id": 1, "file": "a", "source": "X-Wines Slim label image, WineID 9", "url": "u",
                           "licence": "CC0", "matched_by": "m"}], "X-Wines")
    update_sources(path, [{"wine_id": 2, "file": "b", "source": "Open Food Facts: x (y)", "url": "u",
                           "licence": photos.LICENCE, "matched_by": "m"}], "Open Food Facts")
    update_sources(path, [], "X-Wines")          # a new X-Wines run with no match keeps the photo row
    assert pd.read_csv(path)["wine_id"].tolist() == [2]


def make_cfg(tmp_path, monkeypatch):
    root = tmp_path
    (root / "data/processed").mkdir(parents=True)
    (root / "results/demo").mkdir(parents=True)
    pd.DataFrame({"wine_id": [5, 6], "title": ["Georges Duboeuf 2013 Flower Label  (Beaujolais)", "Nobody 2015 Red"],
                  "winery": ["Georges Duboeuf", "Nobody"]}).to_parquet(root / "data/processed/wines.parquet")
    pd.DataFrame({"wine_id": [5, 6, 5]}).to_csv(root / "results/demo/demo_top5.csv", index=False)
    monkeypatch.setattr(photos, "repo_path", lambda p: root / p)
    return {"k": 5, "paths": {"processed_dir": "data/processed"}, "demo": {"out_dir": "results/demo"},
            "images": {"sources": "data/images/sources.csv"},
            "photos": {"extra_ids": "data/images/demo_extra_ids.csv", "candidates_dir": "data/images/candidates",
                       "review": "data/images/photo_review.csv", "photos_dir": "data/images/photos",
                       "pause_seconds": 0}}


def test_find_then_apply_only_the_approved(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, monkeypatch)
    product = {"code": "321", "product_name": "Flower Label", "brands": "Georges Duboeuf", "image_front_url": "http://img"}

    def fake_get(url):
        return {"products": [product] if "duboeuf" in url else []}

    def fake_fetch(url, path):
        Image.new("RGB", (600, 900), "red").save(path, "PNG")

    sheet = photos.find(cfg, get=fake_get, fetch=fake_fetch)
    assert sheet["wine_id"].tolist() == [5, 6]                      # each demo wine once
    assert sheet.set_index("wine_id").loc[6, "photo"] is None or pd.isna(sheet.set_index("wine_id").loc[6, "photo"])
    assert (tmp_path / "data/images/candidates/5.jpg").exists()
    assert not (tmp_path / "data/images/photos").exists()            # nothing used before approval

    review = pd.read_csv(tmp_path / "data/images/photo_review.csv", dtype={"approved": str})
    review.loc[review["wine_id"] == 5, "approved"] = "yes"
    review.to_csv(tmp_path / "data/images/photo_review.csv", index=False)
    rows = photos.apply(cfg)
    assert [r["wine_id"] for r in rows] == [5]
    src = pd.read_csv(tmp_path / "data/images/sources.csv")
    assert src.loc[0, "url"] == "https://world.openfoodfacts.org/product/321"
    assert "CC BY-SA" in src.loc[0, "licence"]
