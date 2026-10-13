"""Run all steps in order: ingest, validate, features, split, train, evaluate.

Run from the project root:   python -m fraud.pipeline.training_pipeline
"""
from pathlib import Path

from fraud.components.data_ingestion import DataIngestion
from fraud.components.data_split import DataSplit
from fraud.components.data_validation import DataValidation
from fraud.components.feature_engineering import FeatureEngineering
from fraud.components.model_evaluation import ModelEvaluation
from fraud.components.model_trainer import ModelTrainer
from fraud.entity.artifact_entity import ModelEvaluationArtifact
from fraud.entity.config_entity import (
    DataIngestionConfig,
    DataSplitConfig,
    DataValidationConfig,
    FeatureEngineeringConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from fraud.exception import FraudException
from fraud.logger import get_logger
from fraud.utils.common import read_yaml

logger = get_logger(__name__)


class TrainingPipeline:
    """Builds every step's config from params.yaml, then runs the steps, passing each artifact on."""

    def __init__(self, params_path: str | Path = "params.yaml"):
        self.params = read_yaml(params_path)

    # ---- configs: the ONLY place that turns params.yaml into config objects ----

    def _ingestion_config(self) -> DataIngestionConfig:
        d = self.params["data"]
        return DataIngestionConfig(
            raw_train_path=Path(d["raw_train"]),
            raw_test_path=Path(d["raw_test"]),
            ingested_train_path=Path(d["ingested_train"]),
            ingested_test_path=Path(d["ingested_test"]),
        )

    def _validation_config(self) -> DataValidationConfig:
        v = self.params["validation"]
        return DataValidationConfig(schema_path=Path(v["schema"]), report_path=Path(v["report"]))

    def _feature_config(self) -> FeatureEngineeringConfig:
        d, f = self.params["data"], self.params["features"]
        return FeatureEngineeringConfig(
            features_train_path=Path(d["features_train"]),
            features_test_path=Path(d["features_test"]),
            feature_spec_path=Path(self.params["artifacts"]["feature_spec"]),
            drop_columns=f["drop"],
            txn_window=f["txn_window"],
            features=f["use"],
            output_columns=f["helpers"] + f["use"] + [d["target"]],
        )

    def _split_config(self) -> DataSplitConfig:
        d = self.params["data"]
        return DataSplitConfig(
            cut_date=d["cut_date"],
            time_column=d["time_column"],
            target=d["target"],
            train_path=Path(d["train"]),
            val_path=Path(d["val"]),
            test_path=Path(d["test"]),
        )

    def _trainer_config(self) -> ModelTrainerConfig:
        a = self.params["artifacts"]
        return ModelTrainerConfig(
            model_params=self.params["model"],
            model_path=Path(a["model"]),
            feature_spec_path=Path(a["feature_spec"]),
            target=self.params["data"]["target"],
        )

    def _evaluation_config(self) -> ModelEvaluationConfig:
        e, a = self.params["evaluation"], self.params["artifacts"]
        return ModelEvaluationConfig(
            cutoff=e["cutoff"],
            alert_cost=e["alert_cost"],
            min_val_pr_auc=e["min_val_pr_auc"],
            metrics_path=Path(a["metrics"]),
            feature_spec_path=Path(a["feature_spec"]),
            target=self.params["data"]["target"],
        )

    # ---- the pipeline: each step's artifact is the next step's input ----

    def run_pipeline(self) -> ModelEvaluationArtifact:
        try:
            logger.info("========== Training pipeline started ==========")

            ingestion = DataIngestion(self._ingestion_config()).initiate_data_ingestion()
            validation = DataValidation(ingestion, self._validation_config()).initiate_data_validation()
            features = FeatureEngineering(ingestion, validation, self._feature_config()).initiate_feature_engineering()
            split = DataSplit(features, self._split_config()).initiate_data_split()
            trainer = ModelTrainer(split, self._trainer_config()).initiate_model_training()
            evaluation = ModelEvaluation(split, trainer, self._evaluation_config()).initiate_model_evaluation()

            logger.info("========== Training pipeline finished: %s ==========", evaluation)
            return evaluation
        except Exception as e:
            logger.exception("Training pipeline failed")
            raise FraudException(e) from e


if __name__ == "__main__":
    print(TrainingPipeline().run_pipeline())
