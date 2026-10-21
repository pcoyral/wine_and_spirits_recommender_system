import pytest

from sensoryrec.metrics import gini


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
