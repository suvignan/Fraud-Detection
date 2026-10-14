"""Model checks on the pipeline's outputs. Skipped when those files don't exist yet (e.g. in CI)."""
from pathlib import Path

import pytest

from fraud.utils.common import load_json, read_yaml

SPEC = Path("models/feature_spec.json")
METRICS = Path("reports/metrics.json")


@pytest.mark.skipif(not SPEC.exists(), reason="run the pipeline first: no feature_spec.json")
def test_feature_spec_has_9_features_and_14_categories():
    spec = load_json(SPEC)
    assert len(spec["features"]) == 9
    assert len(spec["categories"]) == 14
    assert spec["categories"] == sorted(spec["categories"])   # fixed, sorted order -> same codes everywhere


@pytest.mark.skipif(not METRICS.exists(), reason="run the pipeline first: no metrics.json")
def test_val_pr_auc_passes_the_quality_gate():
    metrics = load_json(METRICS)
    minimum = read_yaml("params.yaml")["evaluation"]["min_val_pr_auc"]
    assert metrics["val"]["pr_auc"] >= minimum
