"""Reason codes: the top features that pushed THIS transaction toward fraud, as sentences with context."""
import pandas as pd
import xgboost as xgb

NIGHT_HOURS = {22, 23, 0, 1, 2, 3}          # the high-risk window found in EDA (notebook 01)
AMOUNT_FEATURES = {"amt", "amt_ratio"}      # two views of the same thing -> at most ONE amount reason


def describe(feature: str, row: pd.Series) -> str:
    """Turn one feature into a sentence an analyst or customer understands: always with context."""
    value = row[feature]
    if feature in AMOUNT_FEATURES:
        if pd.isna(row["amt_ratio"]):
            return f"Amount ${row['amt']:,.2f} on a card with no earlier purchases to compare"
        return f"Amount ${row['amt']:,.2f} is {row['amt_ratio']:.1f}x this card's usual spend"
    if pd.isna(value):
        return "No earlier purchase on record for this card"
    if feature == "card_avg_amt_before":
        return f"This card usually spends ${value:,.2f}"
    if feature == "category":
        return f"Purchase category '{value}' is a higher-risk category"
    if feature == "hour":
        night = " (high-risk night window)" if int(value) in NIGHT_HOURS else ""
        return f"Purchase at {int(value):02d}:00{night}"
    if feature == "age":
        return f"Cardholder aged {int(value)}"
    if feature == "secs_since_last":
        minutes = value / 60
        gap = f"{minutes:.0f} min" if minutes < 120 else f"{minutes / 60:.1f} h"
        return f"{gap} since this card's previous purchase"
    if feature == "txn_count_24h":
        return f"{int(value)} purchases on this card in the last 24 h"
    if feature == "is_first_txn":
        return "First purchase seen on this card" if value == 1 else "Card has earlier purchases"
    return f"{feature} = {value}"


def reason_codes(model: xgb.XGBClassifier, X: pd.DataFrame, top_n: int = 3) -> list[str]:
    """SHAP pushes for one row (pred_contribs=True); the top_n that pushed TOWARD fraud, one amount reason max."""
    dmatrix = xgb.DMatrix(X, enable_categorical=True)
    pushes = model.get_booster().predict(dmatrix, pred_contribs=True)[0][:-1]  # last column = base value
    ranked = sorted(zip(X.columns, pushes), key=lambda pair: pair[1], reverse=True)

    reasons, amount_used = [], False
    for feature, push in ranked:
        if push <= 0 or len(reasons) == top_n:
            break
        if feature in AMOUNT_FEATURES:
            if amount_used:
                continue                 # the amount is already explained -> give the slot to the next feature
            amount_used = True
        sentence = describe(feature, X.iloc[0])
        if sentence not in reasons:      # e.g. several empty history features -> say "no earlier purchase" once
            reasons.append(sentence)
    return reasons
