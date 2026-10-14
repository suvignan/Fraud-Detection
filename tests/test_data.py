"""Data checks: validation stops bad data; train / val / test don't overlap in time."""
from pathlib import Path

import pandas as pd
import pytest

from fraud.components.data_validation import DataValidation
from fraud.entity.artifact_entity import DataIngestionArtifact
from fraud.entity.config_entity import DataValidationConfig
from fraud.exception import FraudException


def test_validation_stops_when_a_column_is_missing(tmp_path):
    # A tiny file with only 2 of the 22 contract columns
    df = pd.DataFrame({"amt": [100.0, 200.0], "is_fraud": [0, 1]})
    train_file = tmp_path / "train.parquet"
    test_file = tmp_path / "test.parquet"
    df.to_parquet(train_file, index=False)
    df.to_parquet(test_file, index=False)

    artifact = DataIngestionArtifact(
        ingested_train_path=train_file,
        ingested_test_path=test_file,
        train_rows=len(df),
        test_rows=len(df),
    )
    report = tmp_path / "report.json"
    config = DataValidationConfig(schema_path=Path("config/schema.yaml"), report_path=report)

    with pytest.raises(FraudException, match="missing columns"):
        DataValidation(artifact, config).initiate_data_validation()

    assert report.exists()  # the audit trail is written even when validation fails


SPLIT_FILES = [Path(f"data/processed/{name}.parquet") for name in ("train", "val", "test")]


@pytest.mark.skipif(not all(p.exists() for p in SPLIT_FILES), reason="needs the pipeline's split files")
def test_train_val_test_do_not_overlap_in_time():
    t = "trans_date_trans_time"
    train, val, test = (pd.read_parquet(p, columns=[t])[t] for p in SPLIT_FILES)
    assert train.max() < val.min()   # every train row is before every val row
    assert val.max() < test.min()    # every val row is before every test row
