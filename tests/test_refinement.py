import numpy as np
import pandas as pd
import pytest

from sensoryrec.config import repo_path
from sensoryrec.refinement import (Refiner, choose_alpha, direction_vector, has_terms, is_eligible,
                                   is_success, load_lexicons, shift, summarise, summarise_cheaper,
                                   target_rank, top_k)
from sensoryrec.text import normalise

LEXICONS = load_lexicons(repo_path("data/refinement_lexicons.yaml"))


def stems_overlap(a, b):
    """Two words overlap if one is the start of the other ("sweet" / "sweetness")."""
    return a.startswith(b) or b.startswith(a)


def test_lexicon_lists_do_not_overlap():
    # D5: the words that build a direction never measure it (no shared stem).
    for trait, lex in LEXICONS.items():
        build = {w for term in lex["build"] for w in normalise(term).split()}
        measure = {w for term in lex["measure"] for w in normalise(term).split()}
        clashes = [(b, m) for b in build for m in measure if stems_overlap(b, m)]
        assert clashes == [], f"{trait}: {clashes}"


def test_every_trait_has_both_directions_and_lists():
    assert set(LEXICONS) == {"sweetness", "body", "fruitiness", "oak", "acidity", "tannin"}
    for lex in LEXICONS.values():
        assert lex["build"] and lex["measure"] and lex["more"] and lex["less"]


def test_has_terms_with_negation_guard():
    notes = ["Not sweet at all.", "Sweet and ripe.", "Without oak, but honeyed."]
    assert has_terms(notes, ["sweet"]).tolist() == [False, True, False]
    assert has_terms(notes, ["oak", "honeyed"]).tolist() == [False, False, True]


def test_direction_vector_is_difference_of_means():
    # Notes with the trait point along x, the others along y:
    # mean(with) - mean(without) = (1, 0) - (0, 1) = (1, -1) -> length 1: (0.707, -0.707)
    e = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    d = direction_vector(e, np.array([True, True, False, False]))
    assert d == pytest.approx([1 / np.sqrt(2), -1 / np.sqrt(2)])


def test_shift_moves_towards_or_away_from_the_direction():
    q, d = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    small, big = shift(q, d, 0.1), shift(q, d, 0.5)
    assert np.linalg.norm(big) == pytest.approx(1.0)
    assert small @ d < big @ d                     # larger alpha, closer to d
    assert shift(q, d, 0.5, sign=-1) @ d < 0       # "less" moves away from d
    # by hand: q + 0.5 d = (1, 0.5); length sqrt(1.25) -> (0.894, 0.447)
    assert big == pytest.approx([1 / np.sqrt(1.25), 0.5 / np.sqrt(1.25)])


def test_choose_alpha_takes_smallest_within_tolerance():
    # best = 0.62; within 2 points means >= 0.60; the smallest such alpha is 0.2
    rates = {0.05: 0.40, 0.1: 0.55, 0.2: 0.60, 0.3: 0.61, 0.5: 0.62}
    assert choose_alpha(rates, tolerance_pp=2) == 0.2


def test_success_and_eligibility():
    assert is_success(0.2, 0.4, sign=1) and not is_success(0.4, 0.4, sign=1)
    assert is_success(0.4, 0.2, sign=-1)
    # nothing to remove / no room to grow -> not eligible
    assert not is_eligible(0.0, sign=-1) and not is_eligible(1.0, sign=1)
    assert is_eligible(0.0, sign=1) and is_eligible(1.0, sign=-1)


def test_top_k_and_target_rank_break_ties_by_wine_id():
    scores = np.array([0.5, 0.9, 0.9, 0.1])
    ids = np.array([10, 30, 20, 40])
    assert top_k(scores, ids, 2).tolist() == [2, 1]          # wine 20 before wine 30
    assert target_rank(scores, ids, 1) == 2                  # wine 30 is second
    assert target_rank(scores, ids, 3) == 4


def toy_refiner():
    # Two sweet notes point up (y), two dry notes point right (x).
    catalogue = pd.DataFrame({
        "wine_id": [1, 2, 3, 4],
        "description": ["sweet and honey", "sweet, syrupy", "dry lemon", "crisp lemon"],
        "price": [30.0, 25.0, 10.0, np.nan]})
    e = np.array([[0.0, 1.0], [0.2, 0.98], [1.0, 0.0], [0.98, 0.2]])
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    lex = {"sweetness": {"build": ["sweet"], "measure": ["honey", "syrupy"], "more": "sweeter", "less": "drier"}}
    return Refiner(e, catalogue, lex).fit(holdout_ids=[])


def test_refiner_end_to_end():
    r = toy_refiner()
    # direction = mean(sweet) - mean(dry) points up-left: (-0.707, 0.707)
    assert r.directions["sweetness"] == pytest.approx([-1 / np.sqrt(2), 1 / np.sqrt(2)], abs=0.01)
    # Query = wine 3's direction (1, 0). Top 1 before: wine 3, no measure word.
    # "sweeter" with alpha 1.5: q' ~ (-0.06, 1.0) -> top 1 = wine 1 ("honey") -> success.
    # A random direction along +x keeps wine 3 on top -> no success.
    cases = r.evaluate([np.array([1.0, 0.0])], [3], alpha=1.5, k=1,
                       random_dirs=[np.array([1.0, 0.0])]).set_index("sign")
    more, less = cases.loc[1], cases.loc[-1]
    assert more["share_before"] == 0 and more["share_after"] == 1 and more["success"]
    assert more["random_success"] == 0
    assert more["rank_before"] == 1 and more["rank_after"] == 4   # the cost: wine 3 drops to last
    assert not less["eligible"]       # "drier": nothing sweet in the top 1 to remove


def test_cheaper_records_price_drop_and_relevance_cost():
    r = toy_refiner()
    # Query along y: top 2 = wines 1 ($30, similarity 1) and 2 ($25, similarity 0.98).
    # Median $27.5. Filter <= 27.5: wine 1 out, wine 4 (no price) out -> wines 2 and 3
    # ($25 and $10, similarity 0.98 and 0) -> median $17.5.
    row = r.cheaper([np.array([0.0, 1.0])], [1], k=2).iloc[0]
    sim2 = 0.98 / np.linalg.norm([0.2, 0.98])
    assert row["eligible"] and row["price_drop_pct"] == pytest.approx((27.5 - 17.5) / 27.5 * 100)
    assert row["relevance_before"] == pytest.approx((1 + sim2) / 2)
    assert row["relevance_after"] == pytest.approx((sim2 + 0) / 2)
    assert row["target_in_before"] and not row["target_in_after"]


def test_summarise_cheaper_with_an_ineligible_row():
    # The ineligible row has no target_in_after: the column becomes mixed;
    # the percentage must still be a proper share (the review found -150 %).
    cases = pd.DataFrame([
        {"wine_id": 1, "eligible": False, "target_in_before": False},
        {"wine_id": 2, "eligible": True, "target_in_before": True, "target_in_after": False,
         "price_drop_pct": 20.0, "relevance_before": 0.8, "relevance_after": 0.7},
        {"wine_id": 3, "eligible": True, "target_in_before": True, "target_in_after": True,
         "price_drop_pct": 40.0, "relevance_before": 0.6, "relevance_after": 0.4}])
    row = summarise_cheaper(cases)
    assert row["n_eligible"] == 2 and "success_rate" not in row
    assert row["median_price_drop_pct"] == pytest.approx(30.0)
    assert row["mean_relevance_cost"] == pytest.approx(0.15)      # (0.1 + 0.2) / 2
    assert row["pct_target_left_topk"] == pytest.approx(50.0)     # 1 of the 2 left


def test_summarise():
    cases = pd.DataFrame({
        "trait": ["oak"] * 4, "sign": [1] * 4,
        "eligible": [True, True, True, False], "success": [True, False, True, False],
        "random_success": [0.1, 0.0, 0.2, 0.0],
        "rank_before": [1, 3, 10, 2], "rank_after": [4, 2, 12, 2]})
    row = summarise(cases, {"oak": {"more": "more oaky", "less": "less oaky"}}, k=5).iloc[0]
    assert row["n_eligible"] == 3 and row["success_rate"] == pytest.approx(2 / 3)
    assert row["random_success_rate"] == pytest.approx(0.1)          # (0.1 + 0 + 0.2) / 3
    assert row["median_rank_change"] == pytest.approx(1.0)           # changes 3, -1, 2, 0
    # in the top 5 before: ranks 1, 3, 2 -> after 4, 2, 2: none left the top 5
    assert row["pct_target_left_topk"] == pytest.approx(0.0)
    # with k = 2, only ranks 1 and 2 count as "in the top k": rank 1 -> 4 left
    row2 = summarise(cases, {"oak": {"more": "more oaky", "less": "less oaky"}}, k=2).iloc[0]
    assert row2["n_target_in_topk_before"] == 2 and row2["pct_target_left_topk"] == pytest.approx(50.0)
