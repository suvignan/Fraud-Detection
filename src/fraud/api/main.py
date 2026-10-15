"""FastAPI app: receive a transaction, return a fraud score and reason codes.

Run from the project root:   uvicorn fraud.api.main:app --reload
"""
import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

from fraud.api.card_key import card_key
from fraud.api.db import PredictionLog, create_tables, get_engine
from fraud.api.history import FileHistoryStore, HistoryStore, NeonHistoryStore, add_to_card_state
from fraud.api.reasons import reason_codes
from fraud.api.schemas import PredictionResponse, TransactionRequest
from fraud.features.build_features import add_time_features, card_features_from_totals, make_model_input
from fraud.logger import get_logger
from fraud.utils.common import load_json, read_yaml

logger = get_logger(__name__)


def build_model_input(txn: TransactionRequest, history: HistoryStore, window: str, spec: dict) -> pd.DataFrame:
    """The API's feature path: card running totals -> the 5 card features; shared recipes for the rest."""
    state = history.get_card_state(txn.cc_num, txn.trans_date_trans_time)        # 1. what we know before now
    row = pd.DataFrame([txn.model_dump()])                                        # 2. this transaction
    card_features = card_features_from_totals(                                    # 3. the 5 card features
        state.txn_count, state.amt_total, state.last_txn_time, state.recent_txn_times,
        txn.trans_date_trans_time, txn.amt, window,
    )
    row = row.assign(**card_features)
    row = add_time_features(row)                                                  # 4. hour, age
    return make_model_input(row, spec)                                            # 5. spec order + 14 categories


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: load everything every request needs, keep it on app.state."""
    params = read_yaml("params.yaml")
    model_path = Path(params["artifacts"]["model"])

    model = XGBClassifier()
    model.load_model(model_path)

    app.state.model = model
    app.state.spec = load_json(params["artifacts"]["feature_spec"])
    app.state.cutoff = params["evaluation"]["cutoff"]
    app.state.window = params["features"]["txn_window"]
    app.state.model_version = hashlib.md5(model_path.read_bytes()).hexdigest()[:12]

    # Which history store? Decided by .env, not by code: HISTORY_BACKEND=file (default) or neon
    load_dotenv()
    backend = os.environ.get("HISTORY_BACKEND", "file")
    if backend == "neon":
        app.state.engine = get_engine()
        create_tables(app.state.engine)
        app.state.history = NeonHistoryStore(app.state.engine, app.state.window)
    elif backend == "file":
        app.state.engine = None                       # file mode: offline, nothing is written anywhere
        app.state.history = FileHistoryStore(
            [Path(params["data"]["ingested_train"]), Path(params["data"]["ingested_test"])], app.state.window
        )
    else:
        raise RuntimeError(f"HISTORY_BACKEND must be 'file' or 'neon', got '{backend}'")
    logger.info("History backend: %s", backend)

    logger.info("Model loaded: version %s, cutoff %.2f", app.state.model_version, app.state.cutoff)
    yield  # setup done -> serve requests (anything after this line would run at shutdown)


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Load balancers ping this to check the API is alive before sending traffic."""
    return {"status": "ok", "model_version": app.state.model_version}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest) -> PredictionResponse:
    """Score one transaction: features -> fraud score -> decision at the cutoff -> reasons."""
    X = build_model_input(txn, app.state.history, app.state.window, app.state.spec)
    score = float(app.state.model.predict_proba(X)[0, 1])          # 6. [row 0, fraud column]
    decision = "flag" if score >= app.state.cutoff else "approve"
    response = PredictionResponse(                                  # 7.
        fraud_score=round(score, 4),
        decision=decision,
        threshold=app.state.cutoff,
        model_version=app.state.model_version,
        reason_codes=reason_codes(app.state.model, X),
    )
    if app.state.engine is not None:                                # Neon mode: log + remember this transaction
        record_prediction(app.state.engine, txn, X, response, app.state.window)
    return response


def record_prediction(
    engine, txn: TransactionRequest, X: pd.DataFrame, response: PredictionResponse, window: str
) -> None:
    """One database transaction: the prediction goes in the log AND the purchase updates the card's totals."""
    key = card_key(txn.cc_num)
    features = {name: (None if pd.isna(value) else value if isinstance(value, str) else float(value))
                for name, value in X.iloc[0].items()}           # JSON has no NaN -> empty values become null
    with Session(engine) as session, session.begin():        # both rows saved, or neither
        session.add(PredictionLog(
            model_version=response.model_version, card_key=key, txn_time=txn.trans_date_trans_time,
            amt=txn.amt, category=txn.category, features=features,
            score=response.fraud_score, decision=response.decision,
        ))
        add_to_card_state(session, key, txn.trans_date_trans_time, txn.amt, window)
