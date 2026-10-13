"""Split by time at the cut date into train / val / test; save data/processed."""
from pathlib import Path

import pandas as pd

from fraud.entity.artifact_entity import DataSplitArtifact, FeatureEngineeringArtifact
from fraud.entity.config_entity import DataSplitConfig
from fraud.exception import FraudException
from fraud.logger import get_logger
from fraud.utils.common import read_yaml

logger = get_logger(__name__)


class DataSplit:
    """Cut the train features by time: before the cut date = train, on or after = val. Test is untouched."""

    def __init__(self, feature_artifact: FeatureEngineeringArtifact, config: DataSplitConfig):
        self.feature_artifact = feature_artifact
        self.config = config

    def _save(self, df: pd.DataFrame, path: Path, name: str) -> None:
        """Save one set and log its size and fraud rate."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("%s: %d rows, fraud rate %.3f%%", name, len(df), df[self.config.target].mean() * 100)

    def initiate_data_split(self) -> DataSplitArtifact:
        """Time split, never random: the model always predicts the future."""
        try:
            logger.info("Starting data split at %s", self.config.cut_date)
            time = self.config.time_column
            cut = pd.to_datetime(self.config.cut_date)

            features = pd.read_parquet(self.feature_artifact.features_train_path)
            features = features.sort_values(time).reset_index(drop=True)
            train = features[features[time] < cut].reset_index(drop=True)
            val = features[features[time] >= cut].reset_index(drop=True)
            test = pd.read_parquet(self.feature_artifact.features_test_path)

            if len(train) + len(val) != len(features):
                raise ValueError("Train + val rows don't add up to the original rows")

            self._save(train, self.config.train_path, "train")
            self._save(val, self.config.val_path, "val")
            self._save(test, self.config.test_path, "test")

            return DataSplitArtifact(
                train_path=self.config.train_path,
                val_path=self.config.val_path,
                test_path=self.config.test_path,
                train_rows=len(train),
                val_rows=len(val),
                test_rows=len(test),
            )
        except Exception as e:
            logger.exception("Data split failed")
            raise FraudException(e) from e


if __name__ == "__main__":
    params = read_yaml("params.yaml")
    data = params["data"]

    # Quick test: reuse the feature files already on disk instead of rerunning every earlier step.
    feature_artifact = FeatureEngineeringArtifact(
        features_train_path=Path(data["features_train"]),
        features_test_path=Path(data["features_test"]),
        feature_spec_path=Path(params["artifacts"]["feature_spec"]),
        n_features=len(params["features"]["use"]),
    )
    config = DataSplitConfig(
        cut_date=data["cut_date"],
        time_column=data["time_column"],
        target=data["target"],
        train_path=Path(data["train"]),
        val_path=Path(data["val"]),
        test_path=Path(data["test"]),
    )
    print(DataSplit(feature_artifact, config).initiate_data_split())
