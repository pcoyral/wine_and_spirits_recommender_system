import pandas as pd

from sensoryrec.config import repo_path
from sensoryrec.images import SHAPES, bottle_shape, bottle_svg, load_shapes, match_xwines, vintage


def test_shapes_table_is_valid():
    t = pd.read_csv(repo_path("data/bottle_shapes.csv"))
    assert t["variety"].is_unique and set(t["shape"]) <= set(SHAPES)


def test_bottle_shape():
    table = load_shapes(repo_path("data/bottle_shapes.csv"))
    assert bottle_shape("Merlot", "red", table) == "bordeaux"
    assert bottle_shape("Pinot Noir", "red", table) == "burgundy"
    assert bottle_shape("Riesling", "white", table) == "flute"
    assert bottle_shape("Unknown Grape", "sparkling", table) == "champagne"   # by colour
    assert bottle_shape("Unknown Grape", None, table) == "bordeaux"           # last resort


def test_bottle_svg():
    svg = bottle_svg("flute", "white", height=90)
    assert svg.startswith("<svg") and 'height="90"' in svg and "flute bottle" in svg and "#cbb67c" in svg


def test_vintage():
    assert vintage("Heron Hill 2015 Semi-Sweet Riesling (Finger Lakes)") == 2015
    assert vintage("Karma NV Side View Chardonnay") is None


def test_match_xwines_is_conservative():
    ours = pd.DataFrame({"wine_id": [1, 2, 3, 4, 5],
                         "winery": ["Casa Valduga", "Casa Valduga", "Dr. Loosen", "Nobody", "Casa Valduga"],
                         "title": ["Casa Valduga 2018 Origem Merlot (Serra)", "Casa Valduga 2001 Origem Merlot",
                                   "Dr. Loosen 2015 Wehlener Sonnenuhr Kabinett Riesling (Mosel)", "Nobody 2015 Red",
                                   "Casa Valduga 2018 Raizes Merlot"],
                         "variety": ["Merlot", "Merlot", "Riesling", "Red Blend", "Merlot"]})
    theirs = pd.DataFrame({"WineID": [100062, 106552], "WineryName": ["Casa Valduga", "Dr. Loosen"],
                           "WineName": ["Origem Merlot", "Riesling"],
                           "Grapes": ["['Merlot']", "['Riesling']"],
                           "Vintages": ["[2020, 2019, 2018]", "[2016, 2015]"]})
    # 2: vintage 2001 not listed; 3: the X-Wines name is only a grape, so it could be any
    # Dr. Loosen Riesling; 4: no such winery; 5: "Origem" is missing from the title
    assert match_xwines(ours, theirs) == {1: 100062}


def test_thumbnail(tmp_path):
    from PIL import Image
    from sensoryrec.images import thumbnail
    Image.new("RGB", (900, 1200), "white").save(tmp_path / "big.png")
    thumbnail(tmp_path / "big.png", tmp_path / "small.jpg", height=320)
    with Image.open(tmp_path / "small.jpg") as im:
        assert max(im.size) == 320 and im.format == "JPEG"


def test_match_needs_same_grape_and_ignores_region_words():
    ours = pd.DataFrame({"wine_id": [1, 2, 3, 4],
                         "winery": ["Kendall-Jackson", "Kendall-Jackson", "Duckhorn", "Duckhorn"],
                         "title": ["Kendall-Jackson 2011 Vintner's Reserve Merlot (Sonoma)",
                                   "Kendall-Jackson 2011 Vintner's Reserve Zinfandel (Mendocino)",
                                   "Duckhorn 2013 Sauvignon Blanc (Napa Valley)",
                                   "Duckhorn 2012 Three Palms Merlot (Napa Valley)"],
                         "variety": ["Merlot", "Zinfandel", "Sauvignon Blanc", "Merlot"]})
    theirs = pd.DataFrame({"WineID": [7, 8, 9], "WineryName": ["Kendall-Jackson", "Duckhorn", "Duckhorn"],
                           "WineName": ["Vintner's Reserve Merlot", "Napa Valley Merlot", "Three Palms Merlot"],
                           "Grapes": ["['Merlot']", "['Merlot']", "['Merlot']"],
                           "Vintages": ["[2011]", "[2013, 2012]", "[2012]"],
                           "RegionName": ["Sonoma", "Napa Valley", "Napa Valley"]})
    # 2: Zinfandel is not the Merlot; 3: "Napa Valley" is only the region and the grape differs
    assert match_xwines(ours, theirs) == {1: 7, 4: 9}


def test_match_rejects_a_more_specific_wine():
    ours = pd.DataFrame({"wine_id": [1, 2], "winery": ["Robert Weil", "Robert Weil"],
                         "title": ["Robert Weil 2014 Trocken Riesling (Rheingau)",
                                   "Robert Weil 2014 Kiedrich Turmberg Trocken Riesling (Rheingau)"],
                         "variety": ["Riesling", "Riesling"]})
    theirs = pd.DataFrame({"WineID": [5], "WineryName": ["Robert Weil"], "WineName": ["Trocken Riesling"],
                           "Grapes": ["['Riesling']"], "Vintages": ["[2014]"], "RegionName": ["Rheingau"]})
    assert match_xwines(ours, theirs) == {1: 5}       # 2 is a single-vineyard wine, not this bottle


def test_featured_one_per_bottle_and_varied():
    from sensoryrec.images import featured
    wines = pd.DataFrame({"wine_id": [1, 2, 3, 4, 5], "points": [90, 95, 88, 91, 87],
                          "variety": ["Merlot", "Merlot", "Riesling", "Chardonnay", "Brut"]}).set_index("wine_id")
    sources = pd.DataFrame({"wine_id": [1, 2, 3, 4, 5],
                            "source": ["X-Wines Slim label image, WineID 9", "X-Wines Slim label image, WineID 9",
                                       "X-Wines Slim label image, WineID 7", "X-Wines Slim label image, WineID 6",
                                       "X-Wines Slim label image, WineID 5"]})
    colour = {"Merlot": "red", "Riesling": "white", "Chardonnay": "white", "Brut": "sparkling"}
    picked = featured(sources, wines, colour, n=3, seed=1)
    assert len(picked) == 3
    assert 1 not in picked                       # same bottle as wine 2, which has more points
    assert picked[0] == 2 and 5 in picked        # red first, then white, then sparkling
    assert picked == featured(sources, wines, colour, n=3, seed=1)

