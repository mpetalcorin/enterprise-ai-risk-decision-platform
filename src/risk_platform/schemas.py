from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_amount: float = Field(gt=0, le=1_000_000)
    account_age_days: int = Field(ge=0, le=30_000)
    transactions_24h: int = Field(ge=0, le=10_000)
    avg_amount_30d: float = Field(gt=0, le=1_000_000)
    international: int = Field(ge=0, le=1)
    high_risk_country: int = Field(ge=0, le=1)
    device_new: int = Field(ge=0, le=1)
    failed_logins_24h: int = Field(ge=0, le=1_000)
    transaction_hour: int = Field(ge=0, le=23)
    customer_tenure_years: float = Field(ge=0, le=100)

    @field_validator("international", "high_risk_country", "device_new")
    @classmethod
    def binary_only(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("must be 0 or 1")
        return value


class PredictionRequest(BaseModel):
    transaction: Transaction
    explain: bool = True


class BatchPredictionRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=5000)
    explain: bool = False


class Driver(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    request_id: str
    risk_probability: float
    risk_band: Literal["low", "medium", "high"]
    decision: Literal["approve", "manual_review"]
    threshold: float
    model_backend: str
    model_version: str
    top_drivers: list[Driver] = []


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
