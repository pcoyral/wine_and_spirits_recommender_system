import pandas as pd

from sensoryrec.config import repo_path
from sensoryrec.map_mockup import BANNER, MOCK_SHOPS, origin_counts, origin_figure, page, shops_figure

ISO = pd.Series({"US": "USA", "Portugal": "PRT"})
TOP = pd.DataFrame({"title": ["Wine A", "Wine B", "Wine C", "Wine D"],
                    "country": ["US", "Portugal", "US", "Unknown"]})


def test_origin_counts():
    origins = origin_counts(TOP, ISO).set_index("country")
    # "Unknown" has no code and is left off the map
    assert origins.index.tolist() == ["US", "Portugal"]
    assert origins.loc["US", ["iso3", "n_wines", "wines"]].tolist() == ["USA", 2, "Wine A<br>Wine C"]
    assert origins.loc["Portugal", ["iso3", "n_wines"]].tolist() == ["PRT", 1]


def test_country_code_table():
    iso = pd.read_csv(repo_path("data/country_iso3.csv"))
    assert iso["country"].is_unique and iso["iso3"].is_unique
    assert iso["iso3"].str.fullmatch("[A-Z]{3}").all()
    assert dict(zip(iso["country"], iso["iso3"]))["US"] == "USA"


def test_every_mock_element_is_labelled():
    shops = shops_figure(MOCK_SHOPS)
    assert "MOCK" in shops.layout.title.text
    assert all("MOCK" in name for name in MOCK_SHOPS["name"])   # every pin says so
    origins = origin_figure(origin_counts(TOP, ISO), "something red", "tfidf")
    assert "something red" in origins.layout.title.text and "tfidf" in origins.layout.title.text
    html = page([origins, shops])
    assert BANNER in html and "INVENTED" in BANNER
    assert html.count("cdn.plot.ly/plotly") == 1                # the library is loaded once
