"""Read the raw CSVs from data/raw."""
from pathlib import Path

import pandas as pd

from fraud.entity.artifact_entity import DataIngestionArtifact
from fraud.entity.config_entity import DataIngestionConfig
from fraud.exception import FraudException
from fraud.logger import get_logger
from fraud.utils.common import read_yaml

logger = get_logger(__name__)


class DataIngestion:
    """Get the raw CSVs into the pipeline as clean, typed Parquet files."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read one raw CSV, parse the timestamp, drop the old row-number column."""
        df = pd.read_csv(path)
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")
        return df

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Read, clean and save the train and test files; return where they went."""
        try:
            logger.info("Starting data ingestion")

            train_df = self._read_csv(self.config.raw_train_path)
            test_df = self._read_csv(self.config.raw_test_path)
            logger.info("Train rows: %d", len(train_df))
            logger.info("Test rows: %d", len(test_df))

            self.config.ingested_train_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.ingested_test_path.parent.mkdir(parents=True, exist_ok=True)

            train_df.to_parquet(self.config.ingested_train_path, index=False)
            test_df.to_parquet(self.config.ingested_test_path, index=False)
            logger.info("Saved %s and %s", self.config.ingested_train_path, self.config.ingested_test_path)

            return DataIngestionArtifact(
                ingested_train_path=self.config.ingested_train_path,
                ingested_test_path=self.config.ingested_test_path,
                train_rows=len(train_df),
                test_rows=len(test_df),
            )
        except Exception as e:
            logger.exception("Data ingestion failed")
            raise FraudException(e) from e


if __name__ == "__main__":
    data = read_yaml("params.yaml")["data"]
    config = DataIngestionConfig(
        raw_train_path=Path(data["raw_train"]),
        raw_test_path=Path(data["raw_test"]),
        ingested_train_path=Path(data["ingested_train"]),
        ingested_test_path=Path(data["ingested_test"]),
    )
    artifact = DataIngestion(config).initiate_data_ingestion()
    print(artifact)
