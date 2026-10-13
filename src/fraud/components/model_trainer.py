"""Train the XGBoost champion with early stopping; save the model to models/."""
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from fraud.entity.artifact_entity import DataSplitArtifact, ModelTrainerArtifact
from fraud.entity.config_entity import ModelTrainerConfig
from fraud.exception import FraudException
from fraud.features.build_features import set_category_type
from fraud.logger import get_logger
from fraud.utils.common import load_json, read_yaml

logger = get_logger(__name__)


class ModelTrainer:
    """Train XGBoost on train, early-stop on val, save with XGBoost's own format."""

    def __init__(self, split_artifact: DataSplitArtifact, config: ModelTrainerConfig):
        self.split_artifact = split_artifact
        self.config = config

    def _prepare(self, df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.Series]:
        """X in the spec's feature order with the spec's 14 categories; y = the target."""
        X = df[spec["features"]].copy()
        X = set_category_type(X, spec["categories"])
        y = df[self.config.target]
        return X, y

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training")
            spec = load_json(self.config.feature_spec_path)

            train_df = pd.read_parquet(self.split_artifact.train_path)
            val_df = pd.read_parquet(self.split_artifact.val_path)
            X_train, y_train = self._prepare(train_df, spec)
            X_val, y_val = self._prepare(val_df, spec)

            # Class weight from TRAIN only: normal / fraud
            scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
            logger.info("scale_pos_weight: %.1f", scale_pos_weight)

            model = XGBClassifier(**self.config.model_params, scale_pos_weight=scale_pos_weight)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(self.config.model_path)
            logger.info("Trees kept: %d (best_iteration %d)", model.best_iteration + 1, model.best_iteration)

            return ModelTrainerArtifact(
                model_path=self.config.model_path,
                best_iteration=int(model.best_iteration),
                scale_pos_weight=scale_pos_weight,
            )
        except Exception as e:
            logger.exception("Model training failed")
            raise FraudException(e) from e


if __name__ == "__main__":
    params = read_yaml("params.yaml")
    data = params["data"]

    # Quick test: reuse the split files already on disk.
    split_artifact = DataSplitArtifact(
        train_path=Path(data["train"]),
        val_path=Path(data["val"]),
        test_path=Path(data["test"]),
        train_rows=0, val_rows=0, test_rows=0,   # not needed by the trainer
    )
    config = ModelTrainerConfig(
        model_params=params["model"],
        model_path=Path(params["artifacts"]["model"]),
        feature_spec_path=Path(params["artifacts"]["feature_spec"]),
        target=data["target"],
    )
    print(ModelTrainer(split_artifact, config).initiate_model_training())
