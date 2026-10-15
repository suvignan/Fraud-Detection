"""Guardrails: rules for inputs the model never learned from, so it can't judge them.

Trees don't extrapolate (the largest training fraud was $1,372; above ~$1,200 every amount gets the same score),
and brand-new cards have no history. In those cases the model doesn't say "fraud", it says "I don't know",
so the honest answer is a THIRD decision: review (a human looks at it). Used by the trainer (limits),
the evaluation (how many reviews) and the API (the decision): one copy, no skew.
"""
import pandas as pd

DECISIONS = ("approve", "review", "flag")


def compute_limits(train_amounts: pd.Series) -> dict:
    """The rule limits, from the TRAINING data only (no business limit is available)."""
    return {
        "max_train_amt": float(train_amounts.max()),               # beyond this: truly out of range
        "new_card_amt_p99": float(train_amounts.quantile(0.99)),   # new card above this: unusual
    }


def check_guardrails(amt: float, is_new_card: bool, limits: dict) -> tuple[str, str] | None:
    """Return (rule name, reason sentence) if a rule fires, else None."""
    if amt > limits["max_train_amt"]:
        return ("amount_out_of_range",
                f"Amount ${amt:,.2f} is above anything in the training data "
                f"(max ${limits['max_train_amt']:,.2f}): manual review")
    if is_new_card and amt > limits["new_card_amt_p99"]:
        return ("new_card_high_amount",
                f"New card with no history and an amount above the usual range "
                f"(${limits['new_card_amt_p99']:,.2f}): manual review")
    return None


def decide(score: float, cutoff: float, rule: tuple[str, str] | None) -> str:
    """Model first: a confident fraud is flagged. Otherwise a fired rule means review. Otherwise approve."""
    if score >= cutoff:
        return "flag"
    return "review" if rule else "approve"
