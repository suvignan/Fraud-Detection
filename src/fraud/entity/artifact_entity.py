"""What each step produces (file paths), passed on to the next step."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    """Where ingestion saved the Parquet files, and how many rows each has."""
    ingested_train_path: Path
    ingested_test_path: Path
    train_rows: int
    test_rows: int


@dataclass(frozen=True)
class DataValidationArtifact:
    """Whether the data passed the contract, and where the report is."""
    validation_status: bool
    report_path: Path
    message: str


@dataclass(frozen=True)
class FeatureEngineeringArtifact:
    """Where the features were saved, and where the feature spec is."""
    features_train_path: Path
    features_test_path: Path
    feature_spec_path: Path
    n_features: int
    