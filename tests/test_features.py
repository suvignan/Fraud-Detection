"""Feature invariants: past only, first purchase has empty history, per card.

Tiny hand-made data, so every expected value can be worked out on paper. Run: pytest
"""
import pandas as pd
import pytest

from fraud.features.build_features import add_card_history_features, add_time_features


@pytest.fixture
def two_cards() -> pd.DataFrame:
    """2 cards x 3 purchases, already sorted by card then time (as the function requires).

    Card 1: 00:00 ($10) -> +1 h ($20) -> +25 h ($60)
    Card 2: 00:30 ($100) -> +10 min ($200) -> +30 min ($300)
    """
    return pd.DataFrame({
        "cc_num": [1, 1, 1, 2, 2, 2],
        "trans_date_trans_time": pd.to_datetime([
            "2020-01-01 00:00", "2020-01-01 01:00", "2020-01-02 02:00",
            "2020-01-01 00:30", "2020-01-01 00:40", "2020-01-01 01:10",
        ]),
        "amt": [10.0, 20.0, 60.0, 100.0, 200.0, 300.0],
    })


def test_secs_since_last_is_empty_on_first_purchase_then_equals_the_gap(two_cards):
    out = add_card_history_features(two_cards, "24h")

    # each card's FIRST purchase has no history -> empty, never borrowed from the other card
    assert pd.isna(out.loc[0, "secs_since_last"])
    assert pd.isna(out.loc[3, "secs_since_last"])

    # the others equal the gaps chosen above (seconds)
    assert out.loc[1, "secs_since_last"] == 3600      # 1 hour
    assert out.loc[2, "secs_since_last"] == 90000     # 25 hours
    assert out.loc[4, "secs_since_last"] == 600       # 10 minutes
    assert out.loc[5, "secs_since_last"] == 1800      # 30 minutes


def test_card_avg_amt_before_never_includes_the_current_purchase(two_cards):
    out = add_card_history_features(two_cards, "24h")

    # card 1: [nothing before] -> [10] -> [10, 20]
    assert pd.isna(out.loc[0, "card_avg_amt_before"])
    assert out.loc[1, "card_avg_amt_before"] == 10
    assert out.loc[2, "card_avg_amt_before"] == 15    # (10 + 20) / 2, NOT including the 60

    # card 2 starts fresh: it must not see card 1's purchases
    assert pd.isna(out.loc[3, "card_avg_amt_before"])
    assert out.loc[4, "card_avg_amt_before"] == 100
    assert out.loc[5, "card_avg_amt_before"] == 150   # (100 + 200) / 2, NOT including the 300


def test_txn_count_24h_counts_only_the_last_24_hours_including_this_one(two_cards):
    out = add_card_history_features(two_cards, "24h")
    # card 1's third purchase is 25 h after the previous one -> only itself in the window
    assert list(out["txn_count_24h"]) == [1, 2, 1, 1, 2, 3]


def test_is_first_txn_marks_only_each_cards_first_purchase(two_cards):
    out = add_card_history_features(two_cards, "24h")
    assert list(out["is_first_txn"]) == [1, 0, 0, 1, 0, 0]


def test_amt_ratio_compares_with_the_cards_earlier_average(two_cards):
    out = add_card_history_features(two_cards, "24h")
    ratios = out["amt_ratio"].tolist()
    assert pd.isna(ratios[0])
    assert ratios[1] == 2.0     # 20 / 10
    assert ratios[2] == 4.0     # 60 / 15
    assert pd.isna(ratios[3])
    assert ratios[4] == 2.0     # 200 / 100
    assert ratios[5] == 2.0     # 300 / 150


def test_age_rounds_down_the_day_before_a_birthday():
    one = pd.DataFrame({
        "trans_date_trans_time": pd.to_datetime(["2020-01-01 01:00"]),
        "dob": ["2000-01-02"],          # turns 20 TOMORROW
    })
    out = add_time_features(one)
    assert out.loc[0, "hour"] == 1
    assert out.loc[0, "age"] == 19      # not 20 yet: age is rounded DOWN
