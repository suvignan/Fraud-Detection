"""Score val/test (PR-AUC, ROC-AUC, cutoff table); save reports/metrics.json."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from fraud.entity.artifact_entity import DataSplitArtifact, ModelEvaluationArtifact, ModelTrainerArtifact
from fraud.entity.config_entity import ModelEvaluationConfig
from fraud.exception import FraudException
from fraud.features.build_features import make_model_input
from fraud.logger import get_logger
from fraud.utils.common import load_json, read_yaml, save_json

logger = get_logger(__name__)


class ModelEvaluation:
    """Evaluate the trained model on val and test; the quality gate uses val only."""

    def __init__(
        self,
        split_artifact: DataSplitArtifact,
        trainer_artifact: ModelTrainerArtifact,
        config: ModelEvaluationConfig,
    ):
        self.split_artifact = split_artifact
        self.trainer_artifact = trainer_artifact
        self.config = config

    def _score(self, model: XGBClassifier, df: pd.DataFrame, spec: dict) -> dict:
        """Every number for ONE set: ranking quality, and what happens at the cutoff."""
        X = make_model_input(df, spec)
        y = df[self.config.target]
        scores = model.predict_proba(X)[:, 1]

        flagged = scores >= self.config.cutoff
        fraud_mask = y == 1
        caught_mask = flagged & fraud_mask
        missed_mask = (~flagged) & fraud_mask

        alerts = int(flagged.sum())
        caught = int(caught_mask.sum())
        frauds = int(fraud_mask.sum())
        total_cost = alerts * self.config.alert_cost + df.loc[missed_mask, "amt"].sum()

        return {
            "rows": len(df),
            "fraud_rate_pct": float(round(y.mean() * 100, 3)),
            "pr_auc": float(round(average_precision_score(y, scores), 4)),
            "roc_auc": float(round(roc_auc_score(y, scores), 4)),
            "frauds": frauds,
            "alerts": alerts,
            "caught": caught,
            "false_alarms": alerts - caught,
            "recall_pct": float(round(caught / frauds * 100, 1)) if frauds else 0.0,
            "precision_pct": float(round(caught / alerts * 100, 1)) if alerts else 0.0,
            "total_cost": int(round(total_cost)),
        }

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Score val and test, always save the metrics, then the quality gate (val only)."""
        try:
            logger.info("Starting model evaluation")
            spec = load_json(self.config.feature_spec_path)

            model = XGBClassifier()
            model.load_model(self.trainer_artifact.model_path)

            val_df = pd.read_parquet(self.split_artifact.val_path)
            test_df = pd.read_parquet(self.split_artifact.test_path)

            val_metrics = self._score(model, val_df, spec)
            test_metrics = self._score(model, test_df, spec)

            report = {
                "cutoff": self.config.cutoff,
                "min_val_pr_auc": self.config.min_val_pr_auc,
                "val": val_metrics,
                "test": test_metrics,
            }
            save_json(self.config.metrics_path, report)  # always, even if the gate fails
            logger.info("PR-AUC | val: %.4f | test: %.4f", val_metrics["pr_auc"], test_metrics["pr_auc"])

            # Quality gate on VAL only: accepting a model is a decision, and decisions never use test
            is_accepted = val_metrics["pr_auc"] >= self.config.min_val_pr_auc
            if not is_accepted:
                logger.error(
                    "Model rejected: val PR-AUC %.4f is below the minimum %.4f",
                    val_metrics["pr_auc"],
                    self.config.min_val_pr_auc,
                )
                raise ValueError(
                    f"Val PR-AUC {val_metrics['pr_auc']:.4f} is below the minimum {self.config.min_val_pr_auc:.4f}"
                )

            return ModelEvaluationArtifact(
                metrics_path=self.config.metrics_path,
                val_pr_auc=val_metrics["pr_auc"],
                test_pr_auc=test_metrics["pr_auc"],
                is_model_accepted=True,
            )
        except Exception as e:
            logger.exception("Model evaluation failed")
            raise FraudException(e) from e


if __name__ == "__main__":
    params = read_yaml("params.yaml")
    data, artifacts, evaluation = params["data"], params["artifacts"], params["evaluation"]

    # Quick test: rebuild the two input "receipts" from the files already on disk.
    split_artifact = DataSplitArtifact(
        train_path=Path(data["train"]),
        val_path=Path(data["val"]),
        test_path=Path(data["test"]),
        train_rows=0, val_rows=0, test_rows=0,   # not used by evaluation
    )
    trainer_artifact = ModelTrainerArtifact(
        model_path=Path(artifacts["model"]),
        best_iteration=85,          # not used by evaluation
        scale_pos_weight=170.4,     # not used by evaluation
    )

    # Every value comes from params.yaml, never typed here.
    config = ModelEvaluationConfig(
        cutoff=evaluation["cutoff"],
        alert_cost=evaluation["alert_cost"],
        min_val_pr_auc=evaluation["min_val_pr_auc"],
        metrics_path=Path(artifacts["metrics"]),
        feature_spec_path=Path(artifacts["feature_spec"]),
        target=data["target"],
    )

    print(ModelEvaluation(split_artifact, trainer_artifact, config).initiate_model_evaluation())
