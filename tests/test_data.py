import numpy as np
import pandas as pd
import pytest

from sensoryrec.data import allocate, clean, make_splits, tail_mask, variety_band
from sensoryrec.eda import top_share

LONG = " ".join(["word"] * 25)  # a 25-word note, long enough for test/dev


def tiny_raw():
    return pd.DataFrame({
        "wine_id": [0, 1, 2, 3, 4],
        "description": ["Crisp and fresh.", "Crisp and fresh. ", "Big and bold.", "", "Soft red fruit."],
        "variety": ["Riesling", "Riesling", None, "Merlot", "Merlot"],
        "price": [10.0, 10.0, 20.0, np.nan, np.nan],
        "country": ["US"] * 5, "province": ["CA"] * 5,
        "region_1": ["Napa", "Napa", None, None, None], "region_2": [None] * 5,
        "designation": [None] * 5, "taster_name": ["A", "A", None, "B", None],
        "taster_twitter_handle": [None] * 5, "winery": ["W1", "W1", "W2", "W3", "W3"],
        "title": ["W1 2015 Riesling"] * 5, "points": [88] * 5,
    })


def test_clean_logs_each_step():
    df, steps, _ = clean(tiny_raw())
    removed = dict(zip(steps["step"], steps["rows_removed"]))
    assert removed["drop empty description"] == 1      # row 3
    assert removed["drop duplicate description"] == 1  # row 1 (after stripping spaces)
    assert removed["drop missing variety"] == 1        # row 2
    assert list(df["wine_id"]) == [0, 4]               # original IDs are kept


def test_clean_fills_unknown_but_not_price():
    df, _, _ = clean(tiny_raw())
    row = df[df["wine_id"] == 4].iloc[0]
    assert row["taster_name"] == "Unknown" and row["taster_name_missing"]
    assert np.isnan(row["price"]) and row["price_missing"]
    assert "taster_twitter_handle" not in df.columns


def test_variety_band_labels():
    df = pd.DataFrame({"wine_id": range(12), "variety": ["A"] * 11 + ["B"]})
    bands = variety_band(df, [10, 100, 1000])
    assert bands.iloc[0] == "11-100" and bands.iloc[-1] == "1-10"


def test_tail_mask():
    # W1 has 3 reviews, W2 has 1: with max_reviews=2 only W2's wine is tail
    df = pd.DataFrame({"winery": ["W1", "W1", "W1", "W2"]})
    assert tail_mask(df, "winery", 2).tolist() == [False, False, False, True]


def test_allocate_minimum_then_proportional():
    # 20 to share, minimum 3: base = 3 + 3 + 2 (band c only has 2)
    # 12 left, shared by room 87 and 7 -> 11.1 and 0.9 -> 11 and 1
    quota = allocate(pd.Series({"a": 90, "b": 10, "c": 2}), total=20, minimum=3)
    assert quota.to_dict() == {"a": 14, "b": 4, "c": 2}
    assert quota.sum() == 20


def make_catalogue():
    rng = np.random.default_rng(0)
    varieties = ["Big"] * 1500 + ["Mid"] * 300 + ["Small"] * 50 + [f"Rare{i}" for i in range(60)]
    n = len(varieties)
    return pd.DataFrame({
        "wine_id": np.arange(n),
        "variety": varieties,
        # about 10% of notes are too short to be picked
        "n_words": np.where(rng.random(n) < 0.1, 10, 25),
    })


CFG = {"split": {"test_size": 200, "dev_size": 100, "min_words": 20,
                 "band_edges": [10, 100, 1000], "test_min_per_band": 30, "dev_min_per_band": 15}}


def test_splits_sizes_disjoint_and_eligible():
    cat = make_catalogue()
    test, dev = make_splits(cat, CFG, seed=42)
    assert len(test) == 200 and len(dev) == 100
    assert set(test["wine_id"]).isdisjoint(dev["wine_id"])
    picked = cat[cat["wine_id"].isin(set(test["wine_id"]) | set(dev["wine_id"]))]
    assert (picked["n_words"] >= 20).all()
    assert (test["band"].value_counts() >= 30).all()


def test_splits_are_reproducible():
    cat = make_catalogue()
    a, _ = make_splits(cat, CFG, seed=42)
    b, _ = make_splits(cat, CFG, seed=42)
    assert list(a["wine_id"]) == list(b["wine_id"])


def test_top_share():
    # counts 50, 30, 10, 10: top 10% of 4 items -> ceil(0.4) = 1 item -> 50/100
    assert top_share([10, 50, 10, 30], 0.10) == pytest.approx(0.5)
