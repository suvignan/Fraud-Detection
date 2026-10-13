"""What each step needs as input (paths and settings), read from params.yaml."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """Where ingestion reads the raw CSVs from and where it saves the Parquet files."""
    raw_train_path: Path
    raw_test_path: Path
    ingested_train_path: Path
    ingested_test_path: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """Where the data contract is and where to write the validation report."""
    schema_path: Path
    report_path: Path


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Where to save the features and spec, plus the feature settings from params.yaml."""
    features_train_path: Path
    features_test_path: Path
    feature_spec_path: Path
    drop_columns: list[str]
    txn_window: str
    features: list[str]          # the 9 model features, in order
    output_columns: list[str]    # helpers + features + target, the saved column order



