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


@dataclass(frozen=True)
class DataSplitConfig:
    """The time cut and where to save train / val / test."""
    cut_date: str
    time_column: str
    target: str
    train_path: Path
    val_path: Path
    test_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    """XGBoost settings, where the spec is, and where to save the model."""
    model_params: dict           # the whole `model:` section of params.yaml
    model_path: Path
    feature_spec_path: Path
    target: str
    
