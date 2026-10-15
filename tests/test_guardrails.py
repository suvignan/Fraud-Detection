"""Guardrails: out-of-range inputs go to review, normal ones are untouched, and the model's flag always wins."""
import pandas as pd

from fraud.guardrails import check_guardrails, compute_limits, decide

LIMITS = {"max_train_amt": 28948.90, "new_card_amt_p99": 546.72}


def test_limits_come_from_the_training_amounts():
    limits = compute_limits(pd.Series([1.0, 2.0, 3.0, 100.0]))
    assert limits["max_train_amt"] == 100.0
    assert 3.0 < limits["new_card_amt_p99"] <= 100.0


def test_amount_above_the_training_range_goes_to_review_even_on_a_known_card():
    rule = check_guardrails(90_000, is_new_card=False, limits=LIMITS)
    assert rule[0] == "amount_out_of_range"
    assert decide(score=0.05, cutoff=0.90, rule=rule) == "review"


def test_new_card_with_a_high_amount_goes_to_review():
    rule = check_guardrails(1_000, is_new_card=True, limits=LIMITS)
    assert rule[0] == "new_card_high_amount"
    assert decide(score=0.05, cutoff=0.90, rule=rule) == "review"


def test_normal_purchases_are_untouched():
    assert check_guardrails(1_000, is_new_card=False, limits=LIMITS) is None   # known card, in range
    assert check_guardrails(40, is_new_card=True, limits=LIMITS) is None       # new card, small amount
    assert decide(score=0.05, cutoff=0.90, rule=None) == "approve"


def test_a_confident_model_flag_beats_any_rule():
    rule = check_guardrails(90_000, is_new_card=True, limits=LIMITS)
    assert decide(score=0.95, cutoff=0.90, rule=rule) == "flag"
