import pandas as pd
import pytest

from sensoryrec.xwines import counts_per, date_range, find_files, long_tail_row, ratings_by_type, summary

WINES = pd.DataFrame({"WineID": [1, 2, 3, 4], "Type": ["Red", "Red", "White", "White"],
                      "Country": ["Italy", "Italy", "France", "Spain"], "WineryID": [10, 10, 20, 30]})
# User 7 rated wine 1 twice (two vintages); wine 4 was never rated.
RATINGS = pd.DataFrame({"UserID": [7, 7, 7, 8, 8, 9], "WineID": [1, 1, 2, 1, 3, 1],
                        "Rating": [4.0, 5.0, 3.0, 4.0, 2.0, 4.0]})


def test_counts_per_includes_unrated_wines():
    assert counts_per(RATINGS, "WineID").to_dict() == {1: 4, 2: 1, 3: 1}
    assert counts_per(RATINGS, "WineID", WINES["WineID"]).to_dict() == {1: 4, 2: 1, 3: 1, 4: 0}
    assert counts_per(RATINGS, "UserID").to_dict() == {7: 3, 8: 2, 9: 1}


def test_long_tail_row():
    # Ratings per wine: 4, 1, 1, 0 -> 6 ratings over 4 wines
    row = long_tail_row("d", "wine", "ratings", [4, 1, 1, 0], few=1)
    assert row["n_units"] == 4 and row["n_events"] == 6 and row["max_per_unit"] == 4
    assert row["mean_per_unit"] == 1.5 and row["median_per_unit"] == 1.0
    # Gini of (0, 1, 1, 4): 2 * (1*0 + 2*1 + 3*1 + 4*4) / (4 * 6) - 5/4 = 42/24 - 30/24 = 0.5
    assert row["gini"] == pytest.approx(0.5)
    # The top 1%, 10% and 20% of 4 wines is 1 wine (rounded up): 4 of 6 ratings
    assert row["top_1pct_share"] == row["top_20pct_share"] == pytest.approx(4 / 6)
    assert row["pct_units_with_1"] == 50.0      # two wines with exactly one rating
    assert row["pct_units_le_1"] == 75.0        # ...plus the unrated wine


def test_summary_counts_a_user_wine_pair_once():
    s = summary(WINES, RATINGS).set_index("metric")["value"]
    assert s["wines_listed"] == 4 and s["wines_rated"] == 3 and s["wineries"] == 3 and s["countries"] == 3
    assert s["users"] == 3 and s["ratings"] == 6
    assert s["user_wine_pairs"] == 5            # (7, 1) counted once
    # 5 pairs out of 3 users x 3 rated wines = 9 possible
    assert s["density_pct"] == pytest.approx(5 / 9 * 100)
    assert s["mean_rating"] == pytest.approx(22 / 6) and s["median_rating"] == 4.0
    assert s["pct_ratings_4_or_more"] == pytest.approx(4 / 6 * 100)


def test_ratings_by_type():
    t = ratings_by_type(WINES, RATINGS).set_index("type")
    # Red: wines 1, 2 -> 5 ratings (4, 5, 3, 4, 4); White: wines 3, 4 -> 1 rating (2)
    assert t.loc["Red", ["wines", "ratings"]].tolist() == [2, 5]
    assert t.loc["White", ["wines", "ratings"]].tolist() == [2, 1]
    assert t.loc["Red", "mean_rating"] == pytest.approx(4.0)
    assert t.loc["Red", "pct_of_ratings"] == pytest.approx(5 / 6 * 100)
    assert t.index.tolist() == ["Red", "White"]  # most rated first


def test_find_files_and_date_range(tmp_path):
    with pytest.raises(SystemExit, match="exactly one"):
        find_files(tmp_path)                    # empty folder
    WINES.to_csv(tmp_path / "XWines_100K_wines.csv", index=False)
    RATINGS.assign(Date=["2015-03-01 10:00:00", "2021-12-31 09:00:00", "2012-07-19 12:40:37",
                         "2016-01-01 00:00:00", "2018-05-05 05:05:05", "2020-02-02 02:02:02"]) \
        .to_csv(tmp_path / "XWines_21M_ratings.csv", index=False)
    wines_path, ratings_path = find_files(tmp_path)
    assert wines_path.name == "XWines_100K_wines.csv" and ratings_path.name == "XWines_21M_ratings.csv"
    assert date_range(ratings_path) == ("2012-07-19 12:40:37", "2021-12-31 09:00:00")
    WINES.to_csv(tmp_path / "XWines_Slim_wines.csv", index=False)
    with pytest.raises(SystemExit, match="exactly one"):
        find_files(tmp_path)                    # two versions mixed in one folder
