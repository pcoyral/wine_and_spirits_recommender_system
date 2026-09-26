import pandas as pd
from src.data_loader import label_lvmh, get_lvmh_subset


def test_label_lvmh_flags_known_brands():
    df = pd.DataFrame({
        "winery": ["Moët & Chandon", "Random Winery", "Krug", "Château Nowhere"],
    })
    labeled = label_lvmh(df)
    assert labeled["is_lvmh"].tolist() == [True, False, True, False]


def test_get_lvmh_subset_returns_only_matches():
    df = pd.DataFrame({
        "winery": ["Moët & Chandon", "Random Winery"],
        "title": ["Wine A", "Wine B"],
    })
    subset = get_lvmh_subset(df)
    assert len(subset) == 1
    assert subset.iloc[0]["title"] == "Wine A"
    assert "is_lvmh" not in subset.columns
