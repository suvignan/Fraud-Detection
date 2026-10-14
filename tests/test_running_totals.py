"""Running totals (API) == full-history features (training), row by row. The safety net for separate code."""
import pandas as pd
import pytest

from fraud.features.build_features import add_card_history_features, card_features_from_totals

CARD_FEATURES = ["secs_since_last", "is_first_txn", "card_avg_amt_before", "amt_ratio", "txn_count_24h"]


@pytest.fixture
def two_cards() -> pd.DataFrame:
    """Same shape as test_features.py: 2 cards, gaps of 1 h / 25 h and 10 / 30 min."""
    return pd.DataFrame({
        "cc_num": [1, 1, 1, 2, 2, 2],
        "trans_date_trans_time": pd.to_datetime([
            "2020-01-01 00:00", "2020-01-01 01:00", "2020-01-02 02:00",
            "2020-01-01 00:30", "2020-01-01 00:40", "2020-01-01 01:10",
        ]),
        "amt": [10.0, 20.0, 60.0, 100.0, 200.0, 300.0],
    })


def test_running_totals_give_the_same_card_features_as_full_history(two_cards):
    expected = add_card_history_features(two_cards, "24h")

    for i, row in two_cards.iterrows():
        past = two_cards[(two_cards.cc_num == row.cc_num) & (two_cards.index < i)]   # only earlier purchases
        got = card_features_from_totals(
            txn_count=len(past),
            amt_total=past["amt"].sum(),
            last_txn_time=past["trans_date_trans_time"].max() if len(past) else None,
            recent_txn_times=past["trans_date_trans_time"].tolist(),
            txn_time=row.trans_date_trans_time,
            amt=row.amt,
            window="24h",
        )
        for feature in CARD_FEATURES:
            want = expected.loc[i, feature]
            if pd.isna(want):
                assert pd.isna(got[feature]), (i, feature)
            else:
                assert got[feature] == pytest.approx(want), (i, feature)   # running sums -> tiny rounding
