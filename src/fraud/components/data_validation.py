"""Check the ingested data against config/schema.yaml (columns, types, empty values, rules)."""
from pathlib import Path

import pandas as pd

from fraud.components.data_ingestion import DataIngestion
from fraud.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from fraud.entity.config_entity import DataIngestionConfig, DataValidationConfig
from fraud.exception import FraudException
from fraud.logger import get_logger
from fraud.utils.common import read_yaml, save_json

logger = get_logger(__name__)

# Type word in schema.yaml -> pandas function that answers "is this column that type?"
TYPE_CHECKS = {
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "str": pd.api.types.is_string_dtype,
    "datetime": pd.api.types.is_datetime64_any_dtype,
}


class DataValidation:
    """Compare the ingested files with the data contract; stop the pipeline if they don't match."""

    def __init__(self, ingestion_artifact: DataIngestionArtifact, config: DataValidationConfig):
        self.ingestion_artifact = ingestion_artifact
        self.config = config
        self.schema = read_yaml(self.config.schema_path)

    def _check_columns(self, df: pd.DataFrame) -> list[str]:
        """Missing or extra columns."""
        problems = []
        expected = set(self.schema["columns"])
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        if missing:
            problems.append(f"missing columns: {sorted(missing)}")
        if extra:
            problems.append(f"extra columns: {sorted(extra)}")
        return problems

    def _check_types(self, df: pd.DataFrame) -> list[str]:
        """Each column has the type the schema promises."""
        problems = []
        for column, type_word in self.schema["columns"].items():
            if column not in df.columns:
                continue  # already reported by _check_columns
            is_right_type = TYPE_CHECKS[type_word]
            if not is_right_type(df[column]):
                problems.append(f"{column}: expected {type_word}, got {df[column].dtype}")
        return problems

    def _check_nulls(self, df: pd.DataFrame) -> list[str]:
        """No empty values in the columns the pipeline uses."""
        problems = []
        for column in self.schema["not_null"]:
            if column in df.columns:
                empty = int(df[column].isna().sum())
                if empty > 0:
                    problems.append(f"{column}: {empty} empty values")
        return problems

    def _check_rules(self, df: pd.DataFrame) -> list[str]:
        """Value rules: positive numbers and allowed values."""
        problems = []
        rules = self.schema["rules"]

        for column in rules["positive"]:
            if column in df.columns:
                bad = int((df[column] <= 0).sum())
                if bad > 0:
                    problems.append(f"{column}: {bad} values are not > 0")

        for column, allowed in rules["allowed_values"].items():
            if column in df.columns:
                unexpected = set(df[column].unique()) - set(allowed)
                if unexpected:
                    problems.append(f"{column}: unexpected values {sorted(map(str, unexpected))}")
        return problems

    def _validate(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Run every check on one DataFrame; an empty list means that check passed."""
        return {
            "columns": self._check_columns(df),
            "types": self._check_types(df),
            "nulls": self._check_nulls(df),
            "rules": self._check_rules(df),
        }

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Validate train and test, always save the report, stop the pipeline on any problem."""
        try:
            logger.info("Starting data validation")
            files = {
                "train": self.ingestion_artifact.ingested_train_path,
                "test": self.ingestion_artifact.ingested_test_path,
            }

            report = {name: self._validate(pd.read_parquet(path)) for name, path in files.items()}
            problems = [
                f"{name}: {message}"
                for name, checks in report.items()
                for messages in checks.values()
                for message in messages
            ]
            status = len(problems) == 0
            report["validation_status"] = status

            save_json(self.config.report_path, report)  # always: passed or failed

            if not status:
                for problem in problems:
                    logger.error("Validation problem - %s", problem)
                raise ValueError(f"Data validation failed: {problems}")

            logger.info("Data validation passed")
            return DataValidationArtifact(
                validation_status=True,
                report_path=self.config.report_path,
                message="All checks passed",
            )
        except Exception as e:
            raise FraudException(e) from e


if __name__ == "__main__":
    params = read_yaml("params.yaml")
    data = params["data"]

    ingestion_config = DataIngestionConfig(
        raw_train_path=Path(data["raw_train"]),
        raw_test_path=Path(data["raw_test"]),
        ingested_train_path=Path(data["ingested_train"]),
        ingested_test_path=Path(data["ingested_test"]),
    )
    ingestion_artifact = DataIngestion(ingestion_config).initiate_data_ingestion()

    validation_config = DataValidationConfig(
        schema_path=Path(params["validation"]["schema"]),
        report_path=Path(params["validation"]["report"]),
    )
    validation_artifact = DataValidation(ingestion_artifact, validation_config).initiate_data_validation()
    print(validation_artifact)
