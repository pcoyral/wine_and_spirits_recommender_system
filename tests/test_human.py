import pandas as pd

from sensoryrec.human import select_human_wines


def test_select_human_wines():
    # 200 test wines in 4 bands; pick 30 with at least 5 per band
    test_ids = pd.DataFrame({"wine_id": range(200),
                             "band": ["1-10"] * 31 + ["11-100"] * 34 + ["101-1000"] * 44 + [">1000"] * 91})
    chosen = select_human_wines(test_ids, n=30, minimum=5, seed=42)
    assert len(chosen) == 30 and chosen["wine_id"].is_unique
    assert set(chosen["wine_id"]) <= set(test_ids["wine_id"])
    assert (chosen["band"].value_counts() >= 5).all()
    # same seed, same wines in the same order
    again = select_human_wines(test_ids, n=30, minimum=5, seed=42)
    assert chosen["wine_id"].tolist() == again["wine_id"].tolist()
