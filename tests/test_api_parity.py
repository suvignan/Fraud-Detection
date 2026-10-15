"""Parity: the API (running totals) builds the same model input as the training pipeline (no skew)."""
from pathlib import Path

import pandas as pd
import pytest

NEEDED = [Path("data/interim/features_test.parquet"), Path("data/interim/ingested_test.parquet"),
          Path("data/interim/ingested_train.parquet"), Path("models/feature_spec.json")]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="needs the pipeline's outputs")
def test_api_features_match_the_pipeline_exactly():
    # imported HERE, not at the top: fraud.api.schemas reads models/feature_spec.json on import,
    # which doesn't exist in CI -> a top-level import would crash before skipif could skip
    from fraud.api.history import FileHistoryStore
    from fraud.api.main import build_model_input
    from fraud.api.schemas import TransactionRequest
    from fraud.features.build_features import make_model_input
    from fraud.utils.common import load_json, read_yaml

    params = read_yaml("params.yaml")
    spec = load_json(params["artifacts"]["feature_spec"])
    window = params["features"]["txn_window"]
    store = FileHistoryStore([Path(params["data"]["ingested_train"]), Path(params["data"]["ingested_test"])], window)

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
        api_X = build_model_input(txn, store, window, spec).reset_index(drop=True)
        pipeline_X = make_model_input(expected.to_frame().T, spec).reset_index(drop=True)
        # running totals add up in a different order -> allow tiny decimal differences (65.0200000001 vs 65.02)
        pd.testing.assert_frame_equal(api_X, pipeline_X.astype(api_X.dtypes), check_exact=False, rtol=1e-9)
