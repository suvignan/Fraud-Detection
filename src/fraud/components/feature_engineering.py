"""Build the 9 features per card, past only (point-in-time); save data/interim."""
from pathlib import Path

import pandas as pd

from fraud.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    FeatureEngineeringArtifact,
)
from fraud.entity.config_entity import FeatureEngineeringConfig
from fraud.exception import FraudException
from fraud.features.build_features import (
    add_card_history_features,
    add_time_features,
    drop_columns,
    set_category_type,
)
from fraud.logger import get_logger
from fraud.utils.common import load_json, read_yaml, save_json

logger = get_logger(__name__)


class FeatureEngineering:
    """Thin manager: reads, combines and sorts the data, calls the shared recipes, saves the results."""

    def __init__(
        self,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
        config: FeatureEngineeringConfig,
    ):
        self.ingestion_artifact = ingestion_artifact
        self.validation_artifact = validation_artifact
        self.config = config

    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        """Build features on train + test together (so test cards keep their train history)."""
        try:
            logger.info("Starting feature engineering")
            if not self.validation_artifact.validation_status:
                raise ValueError("Data validation failed, so features will not be built")

            # 1. Read, label, combine, sort by card then time
            train = pd.read_parquet(self.ingestion_artifact.ingested_train_path)
            test = pd.read_parquet(self.ingestion_artifact.ingested_test_path)
            train["dataset"] = "train"
            test["dataset"] = "test"
            data = pd.concat([train, test], ignore_index=True)
            data = data.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

            # 2. The shared recipes (the same functions the API will use)
            data = add_time_features(data)
            data = add_card_history_features(data, self.config.txn_window)
            data = drop_columns(data, self.config.drop_columns)

            # 3. Split back, keep only the output columns in a fixed order
            is_train = data["dataset"] == "train"
            train = data.loc[is_train, self.config.output_columns].reset_index(drop=True)
            test = data.loc[~is_train, self.config.output_columns].reset_index(drop=True)

            # 4. Categories: TRAIN decides the list and order, test follows
            categories = sorted(train["category"].unique())
            train = set_category_type(train, categories)
            test = set_category_type(test, categories)

            # 5. Save the features and the spec the API needs
            self.config.features_train_path.parent.mkdir(parents=True, exist_ok=True)
            train.to_parquet(self.config.features_train_path, index=False)
            test.to_parquet(self.config.features_test_path, index=False)
            save_json(
                self.config.feature_spec_path,
                {"features": self.config.features, "categories": categories},
            )
            logger.info("Features built: train %d rows, test %d rows", len(train), len(test))

            return FeatureEngineeringArtifact(
                features_train_path=self.config.features_train_path,
                features_test_path=self.config.features_test_path,
                feature_spec_path=self.config.feature_spec_path,
                n_features=len(self.config.features),
            )
        except Exception as e:
            logger.exception("Feature engineering failed")
            raise FraudException(e) from e


if __name__ == "__main__":
    params = read_yaml("params.yaml")
    data, feats = params["data"], params["features"]

    # Standalone (DVC runs each stage as its own command): rebuild both input receipts from disk.
    ingestion_artifact = DataIngestionArtifact(
        ingested_train_path=Path(data["ingested_train"]),
        ingested_test_path=Path(data["ingested_test"]),
        train_rows=0, test_rows=0,   # not used by feature engineering
    )
    # Validation's verdict lives on disk in its report: that file IS the receipt between commands.
    report_path = Path(params["validation"]["report"])
    validation_artifact = DataValidationArtifact(
        validation_status=load_json(report_path)["validation_status"],
        report_path=report_path,
        message="read from the validation report",
    )

    feature_config = FeatureEngineeringConfig(
        features_train_path=Path(data["features_train"]),
        features_test_path=Path(data["features_test"]),
        feature_spec_path=Path(params["artifacts"]["feature_spec"]),
        drop_columns=feats["drop"],
        txn_window=feats["txn_window"],
        features=feats["use"],
        output_columns=feats["helpers"] + feats["use"] + [data["target"]],
    )
    print(FeatureEngineering(ingestion_artifact, validation_artifact, feature_config).initiate_feature_engineering())
