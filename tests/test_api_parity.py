"""Parity: the API builds EXACTLY the same model input as the training pipeline (no training-serving skew)."""
from pathlib import Path

import pandas as pd
import pytest

from fraud.api.history import FileHistoryStore
from fraud.api.main import build_model_input
from fraud.api.schemas import TransactionRequest
from fraud.features.build_features import make_model_input
from fraud.utils.common import load_json, read_yaml

NEEDED = [Path("data/interim/features_test.parquet"), Path("data/interim/ingested_test.parquet"),
          Path("data/interim/ingested_train.parquet"), Path("models/feature_spec.json")]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="needs the pipeline's outputs")
def test_api_features_match_the_pipeline_exactly():
    params = read_yaml("params.yaml")
    spec = load_json(params["artifacts"]["feature_spec"])
    store = FileHistoryStore([Path(params["data"]["ingested_train"]), Path(params["data"]["ingested_test"])])

    # 5 test-period transactions, incl. frauds, with the pipeline's features (the answer key)
    pipeline = pd.read_parquet("data/interim/features_test.parquet")
    sample = pd.concat([pipeline[pipeline.is_fraud == 1].sample(2, random_state=0),
                        pipeline[pipeline.is_fraud == 0].sample(3, random_state=0)])

    # the raw request fields (dob isn't in the features file) come from the ingested file
    keys = ["cc_num", "trans_date_trans_time", "amt"]
    raw = pd.read_parquet("data/interim/ingested_test.parquet", columns=keys + ["category", "dob"])
    requests = sample[keys].merge(raw, on=keys, how="left")

    for (_, req), (_, expected) in zip(requests.iterrows(), sample.iterrows()):
        txn = TransactionRequest(**req.to_dict())
        api_X = build_model_input(txn, store, params["features"]["txn_window"], spec).reset_index(drop=True)
        pipeline_X = make_model_input(expected.to_frame().T, spec).reset_index(drop=True)
        pd.testing.assert_frame_equal(api_X, pipeline_X.astype(api_X.dtypes))
