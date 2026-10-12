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

