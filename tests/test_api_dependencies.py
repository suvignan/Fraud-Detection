"""The API's lean requirements really are enough to START the API (CI installs requirements-api.txt).

Importing a module isn't enough: XGBClassifier imports fine without scikit-learn but fails when CREATED.
That's exactly how the first Render deploy failed, so this test creates the objects the startup creates.
"""
from xgboost import XGBClassifier


def test_the_model_object_can_be_created():
    XGBClassifier()   # needs scikit-learn installed (XGBoost's scikit-learn-style wrapper)
