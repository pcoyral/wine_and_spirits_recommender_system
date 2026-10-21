"""Evaluation metrics. S0 only needs gini(); the rest arrives in S1."""
import numpy as np


def gini(counts):
    """Gini coefficient of a list of non-negative counts.

    0 = every item has the same count; close to 1 = one item has everything.
    Formula on counts sorted ascending x_1..x_n (i = 1..n):
        G = 2 * sum(i * x_i) / (n * sum(x)) - (n + 1) / n
    """
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float(2 * np.sum(i * x) / (n * x.sum()) - (n + 1) / n)
