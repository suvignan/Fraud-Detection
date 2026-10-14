"""Request and response shapes for the API: bad input is rejected (HTTP 422) before it reaches the model."""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from fraud.utils.common import load_json, read_yaml

# The 14 allowed categories come from the SAME spec the model was trained with (one source of truth).
CATEGORIES = load_json(read_yaml("params.yaml")["artifacts"]["feature_spec"])["categories"]


class TransactionRequest(BaseModel):
    """One live card transaction."""
    cc_num: int
    trans_date_trans_time: datetime
    amt: float = Field(gt=0, description="Transaction amount, must be greater than 0")
    category: str
    dob: date

    @field_validator("category")
    @classmethod
    def category_must_be_known(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"unknown category '{value}', must be one of {CATEGORIES}")
        return value

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "cc_num": 4613314721966,
                "trans_date_trans_time": "2020-06-21 22:45:00",
                "amt": 950.0,
                "category": "shopping_net",
                "dob": "1975-03-14",
            }]
        }
    }


class PredictionResponse(BaseModel):
    """The model's verdict, with the reasons behind it."""
    fraud_score: float
    decision: Literal["flag", "approve"]
    threshold: float
    model_version: str
    reason_codes: list[str]
