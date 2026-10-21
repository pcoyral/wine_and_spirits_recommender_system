import numpy as np
import pytest

from sensoryrec.metrics import (
    catalogue_coverage,
    gini,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    recommendation_gini,
    tail_share,
)


def test_gini_equal_counts_is_zero():
    # Everyone has the same: no inequality.
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0)


def test_gini_one_item_has_everything():
    # x = [0, 0, 0, 1], n = 4: 2 * (4 * 1) / (4 * 1) - 5/4 = 2 - 1.25 = 0.75,
    # which is the maximum (n - 1) / n for 4 items.
    assert gini([0, 0, 1, 0]) == pytest.approx(0.75)


def test_gini_hand_example():
    # x = [1, 2, 3], n = 3, sum = 6
    # sum(i * x_i) = 1*1 + 2*2 + 3*3 = 14
    # G = 2 * 14 / (3 * 6) - 4/3 = 28/18 - 24/18 = 4/18 = 0.2222
    assert gini([3, 1, 2]) == pytest.approx(4 / 18)


def test_gini_empty_is_zero():
    assert gini([]) == 0.0


# Two queries: the first target (2) is at rank 2, the second (9) is missing.
RECS = [[1, 2, 3], [4, 5, 6]]
TARGETS = [2, 9]


def test_hit_rate():
    # k=3: query 1 hits, query 2 misses -> 1/2
    assert hit_rate_at_k(RECS, TARGETS, k=3) == pytest.approx(0.5)
    # k=1: wine 2 is at rank 2, outside the top 1 -> 0/2
    assert hit_rate_at_k(RECS, TARGETS, k=1) == pytest.approx(0.0)


def test_ndcg():
    # Three queries: target at rank 1, at rank 3, and missing.
    recs = [[7, 8, 9], [1, 2, 3], [4, 5, 6]]
    targets = [7, 3, 10]
    # rank 1: 1/log2(2) = 1;  rank 3: 1/log2(4) = 1/2;  missing: 0
    # mean = (1 + 0.5 + 0) / 3 = 0.5
    assert ndcg_at_k(recs, targets, k=3) == pytest.approx(0.5)
    # k=2: the rank-3 hit no longer counts -> (1 + 0 + 0) / 3
    assert ndcg_at_k(recs, targets, k=2) == pytest.approx(1 / 3)


def test_ndcg_rank_2():
    # rank 2: 1/log2(3) = 0.6309
    assert ndcg_at_k([[5, 6]], [6], k=2) == pytest.approx(1 / np.log2(3))


def test_catalogue_coverage():
    # Distinct wines recommended: {1, 2, 3} = 3 of a 10-wine catalogue
    assert catalogue_coverage([[1, 2], [2, 3]], catalogue_size=10) == pytest.approx(0.3)


def test_tail_share():
    # 4 slots (1, 2, 2, 3); tail wines are {2, 5}; wine 2 fills 2 slots -> 2/4
    assert tail_share([[1, 2], [2, 3]], tail_ids={2, 5}) == pytest.approx(0.5)


def test_recommendation_gini_popularity_like():
    # Same two wines for both queries, catalogue of 4.
    # counts: wine 1 -> 2, wine 2 -> 2, wines 3, 4 -> 0  => sorted [0, 0, 2, 2]
    # sum(i * x_i) = 3*2 + 4*2 = 14;  G = 2*14 / (4*4) - 5/4 = 1.75 - 1.25 = 0.5
    assert recommendation_gini([[1, 2], [1, 2]], all_items=[1, 2, 3, 4]) == pytest.approx(0.5)


def test_recommendation_gini_spread_out():
    # Every wine recommended once: perfectly even -> 0
    assert recommendation_gini([[1, 2], [3, 4]], all_items=[1, 2, 3, 4]) == pytest.approx(0.0)


def test_novelty():
    # Winery shares: wine 1 -> 1/2, wine 2 -> 1/4, wine 3 -> 1/8
    # Self-information -log2(p): 1, 2 and 3 bits.
    p = {1: 0.5, 2: 0.25, 3: 0.125}
    # list [1, 2]: (1 + 2) / 2 = 1.5;  list [2, 3]: (2 + 3) / 2 = 2.5;  mean = 2.0
    assert novelty([[1, 2], [2, 3]], p) == pytest.approx(2.0)


def test_intra_list_diversity():
    # Wines 1 and 3 point the same way, wine 2 is at a right angle.
    v = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0]), 3: np.array([2.0, 0.0])}
    # [1, 2]: cos = 0 -> ILD 1.   [1, 3]: cos = 1 -> ILD 0.   mean = 0.5
    assert intra_list_diversity([[1, 2], [1, 3]], v) == pytest.approx(0.5)
    # [1, 2, 3]: pairs (1,2) = 0, (1,3) = 1, (2,3) = 0 -> mean 1/3 -> ILD 2/3
    assert intra_list_diversity([[1, 2, 3]], v) == pytest.approx(2 / 3)
