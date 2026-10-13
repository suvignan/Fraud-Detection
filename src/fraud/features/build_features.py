"""Feature recipes shared by training and the API. Pure functions: DataFrame in, DataFrame out."""
import numpy as np
import pandas as pd

TIME = "trans_date_trans_time"
CARD = "cc_num"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour of day and the cardholder's age (whole years) at the time of purchase."""
    df = df.copy()
    df["hour"] = df[TIME].dt.hour
    elapsed_days = (df[TIME] - pd.to_datetime(df["dob"])).dt.days
    df["age"] = np.floor(elapsed_days / 365.25).astype("int64")
    return df


def add_card_history_features(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """Add the 5 card-history features using only each card's PAST purchases.

    The data must already be sorted by card, then time.
    """
    df = df.copy()
    card = df.groupby(CARD)

    df["secs_since_last"] = card[TIME].diff().dt.total_seconds()
    df["is_first_txn"] = df["secs_since_last"].isna().astype("int64")

    # Average of EARLIER purchases only: (running total - this amount) / number of earlier purchases
    df["card_avg_amt_before"] = (card["amt"].cumsum() - df["amt"]) / card.cumcount()
    df["amt_ratio"] = df["amt"] / df["card_avg_amt_before"]

    df["txn_count_24h"] = (
        df.groupby(CARD, sort=False)
        .rolling(window, on=TIME)["amt"]
        .count()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )
    return df


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove the listed columns (ignores any that are already gone)."""
    return df.drop(columns=columns, errors="ignore")


def set_category_type(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Make 'category' a category type with a FIXED order, so train, test and the API agree."""
    df = df.copy()
    df["category"] = pd.Categorical(df["category"], categories=categories)
    return df


def make_model_input(df:pd.DataFrame,spec:dict) -> pd.DataFrame:
    """Return the model's X in the spec's feature order with the spec's 14 categories."""
    X = df[spec["features"]].copy()
    X = set_category_type(X, spec["categories"])
    return X