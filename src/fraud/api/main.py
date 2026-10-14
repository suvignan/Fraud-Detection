"""FastAPI app: receive a transaction, return a fraud score and reason codes.

Run from the project root:   uvicorn fraud.api.main:app --reload
"""
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from xgboost import XGBClassifier

from fraud.api.history import FileHistoryStore, HistoryStore
from fraud.api.reasons import reason_codes
from fraud.api.schemas import PredictionResponse, TransactionRequest
from fraud.features.build_features import add_card_history_features, add_time_features, make_model_input
from fraud.logger import get_logger
from fraud.utils.common import load_json, read_yaml

logger = get_logger(__name__)


def build_model_input(txn: TransactionRequest, history: HistoryStore, window: str, spec: dict) -> pd.DataFrame:
    """The API's feature path: SAME recipe functions as training -> no training-serving skew."""
    past = history.get_history(txn.cc_num, txn.trans_date_trans_time)        # 1. earlier purchases only
    new_row = pd.DataFrame([txn.model_dump()])                                # 2. this transaction...
    combined = pd.concat([past, new_row], ignore_index=True)                  #    ...as the newest row
    combined = add_card_history_features(combined, window)                    # 3. card features
    last_row = add_time_features(combined.tail(1))                            # 4. hour/age need only this row
    return make_model_input(last_row, spec)                                   # 5. spec order + 14 categories


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
    app.state.history = FileHistoryStore([Path(params["data"]["ingested_train"]), Path(params["data"]["ingested_test"])])

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

    return PredictionResponse(                                      # 7.
        fraud_score=round(score, 4),
        decision=decision,
        threshold=app.state.cutoff,
        model_version=app.state.model_version,
        reason_codes=reason_codes(app.state.model, X),
    )
