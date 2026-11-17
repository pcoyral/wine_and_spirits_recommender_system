import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from sensoryrec.xwines_cf import (Content, ItemKNN, Popularity, PureSVD, evaluate, group_of, head_items, read_sampled,
                                  latest_per_pair, leave_last_out, like_matrix, sample_users, top_k, wine_text)


def ratings():
    # user 1 rated wine 10 twice (two vintages): only the 2021 rating counts
    return pd.DataFrame({"UserID": [1, 1, 1, 2, 2, 3],
                         "WineID": [10, 10, 11, 10, 12, 11],
                         "Rating": [2.0, 5.0, 4.0, 4.5, 3.0, 5.0],
                         "Date": ["2019-01-01", "2021-01-01", "2020-01-01", "2020-05-05", "2020-06-06", "2018-01-01"]})


def test_latest_per_pair_and_split():
    r = latest_per_pair(ratings())
    assert len(r) == 5
    assert r[(r.UserID == 1) & (r.WineID == 10)]["Rating"].item() == 5.0
    train, test = leave_last_out(r)
    assert test.set_index("UserID")["WineID"].to_dict() == {1: 10, 2: 12, 3: 11}   # each user's latest
    assert len(train) == 2 and set(train["UserID"]) == {1, 2}


def test_sample_users_is_seeded_and_respects_minimum():
    r = latest_per_pair(ratings())
    s = sample_users(r, min_ratings=2, n_users=5, seed=0)
    assert set(s["UserID"]) == {1, 2}                       # user 3 has a single rating
    assert s.equals(sample_users(r, min_ratings=2, n_users=5, seed=0))


def test_like_matrix_and_head():
    train = pd.DataFrame({"u": [0, 0, 1, 2], "i": [0, 1, 0, 2], "Rating": [4.0, 3.5, 5.0, 4.0]})
    X = like_matrix(train, 4.0, 3, 3)
    assert X.toarray().tolist() == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]       # 3.5 is not a like
    # likes per wine 2, 0, 1 -> wine 0 alone holds 2/3 < 80%, so the head is wines 0 and 2
    assert head_items(np.array([2, 0, 1]), 0.8) == {0, 2}


def test_top_k_skips_rated_wines():
    scores = np.array([[0.9, 0.8, 0.1, 0.5]])
    rated = sparse.csr_matrix([[1, 0, 0, 0]])
    assert top_k(scores, rated, 2).tolist() == [[1, 3]]


#   wine:  0  1  2  3
X = sparse.csr_matrix(np.array([[1, 1, 0, 0],    # user 0
                                [1, 1, 0, 0],    # user 1
                                [0, 1, 1, 0],    # user 2
                                [0, 0, 0, 1]], dtype=np.float32))


def test_item_knn_by_hand():
    # Column norms: wine0 sqrt2, wine1 sqrt3, wine2 1, wine3 1.
    # cos(0,1) = 2/(sqrt2 sqrt3) = 0.816, cos(0,2) = 0, cos(1,2) = 1/sqrt3 = 0.577, cos(0,0)=1
    # A user who liked only wine 0 scores wine j by cos(0, j): [1, 0.816, 0, 0]
    Xq = sparse.vstack([X, sparse.csr_matrix([[1, 0, 0, 0]], dtype=np.float32)]).tocsr()
    m = ItemKNN().fit(Xq)
    s = m.scores(Xq, np.array([4]))[0]
    cos = lambda a, b: (Xq[:, a].T @ Xq[:, b]).toarray().item() / np.sqrt(Xq[:, a].sum() * Xq[:, b].sum())
    assert s == pytest.approx([1.0, cos(0, 1), 0.0, 0.0], abs=1e-5)


def test_popularity_and_svd_and_content():
    assert Popularity().fit(X).scores(X, np.array([0, 1]))[1].tolist() == [2, 3, 1, 1]
    svd = PureSVD(n_factors=2, seed=0).fit(X)
    s = svd.scores(X, np.array([2]))[0]
    assert s[1] > s[3]                                 # wine 1 shares users with user 2's taste; wine 3 does not
    text = ["type=red grape=merlot", "type=red grape=merlot", "type=white grape=riesling", "type=white grape=chardonnay"]
    c = Content().fit(X, text).scores(X, np.array([3]))[0]   # user 3 liked wine 3 (white chardonnay)
    assert c[3] == pytest.approx(1.0) and c[2] > c[0] == pytest.approx(0.0)


def test_wine_text():
    wines = pd.DataFrame({"Type": ["Red"], "Grapes": ["['Merlot', 'Cabernet Franc']"], "Harmonize": ["['Beef']"],
                          "Body": ["Full-bodied"], "Acidity": ["Medium"], "Country": ["France"], "RegionName": [None]})
    assert wine_text(wines) == ["type=red grape=merlot grape=cabernet_franc food=beef body=full-bodied "
                                "acidity=medium country=france"]


def test_groups():
    assert group_of(4, [9, 29, np.inf], ["a", "b", "c"]) == "a"
    assert group_of(30, [9, 29, np.inf], ["a", "b", "c"]) == "c"
    assert group_of(0, [0, 9, 99, np.inf], ["cold", "1-9", "10-99", "100+"]) == "cold"


def test_evaluate_end_to_end():
    rated = X.copy()
    targets = {0: 2, 3: 0}                 # user 0 should find wine 2, user 3 wine 0
    like_counts = np.asarray(X.sum(axis=0)).ravel()
    summary, per_user = evaluate([Popularity().fit(X)], X, rated, targets, np.array([2, 2, 2, 1]),
                                 like_counts, head={1}, k_values=[1, 2])
    s = summary.set_index("k")
    # popularity after removing rated wines: user 0 -> [2 or 3 (tie, by position) ...] = [2, 3]; user 3 -> [1, 0]
    assert s.loc[1, "hr"] == 0.5 and s.loc[2, "hr"] == 1.0
    assert set(per_user["item_group"]) <= {"1-9", "0 likes (cold)"}


def test_read_sampled_matches_the_in_memory_version(tmp_path):
    path = tmp_path / "ratings.csv"
    ratings().to_csv(path, index=False)
    a = read_sampled(path, min_ratings=2, n_users=5, seed=0, chunksize=2)
    b = latest_per_pair(sample_users(latest_per_pair(ratings()), 2, 5, 0))
    pd.testing.assert_frame_equal(a.sort_values(["UserID", "WineID"]).reset_index(drop=True),
                                  b.sort_values(["UserID", "WineID"]).reset_index(drop=True),
                                  check_dtype=False)   # the file reader uses int32
