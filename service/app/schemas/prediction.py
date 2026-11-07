"""HTTP contract. `extra="forbid"` is deliberate: unknown fields, e.g. the leaky
`recovery_flag_after_90d`, are rejected rather than silently ignored (validation condition R1)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_EXAMPLE = {
    "application_id": "APP-2026-000123",
    "loan_amount": 12000,
    "income": 24000,
    "past_default": 0,
    "past_due_30": 1,
    "past_due_60": 0,
    "past_due_90": 0,
    "region_risk": 1,
    "credit_utilization": 0.35,
}


class LoanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_EXAMPLE]})

    application_id: str | None = Field(default=None, max_length=64, description="Caller's reference, echoed back")
    loan_amount: float = Field(gt=0, le=80_000, description="Requested loan amount (product maximum 80,000)")
    income: float = Field(gt=0, le=10_000_000, description="Monthly income")
    past_default: Literal[0, 1] = Field(description="Historical default indicator")
    past_due_30: int = Field(ge=0, le=50, description="Number of loans that reached 30+ DPD")
    past_due_60: int = Field(ge=0, le=50, description="Number of loans that reached 60+ DPD")
    past_due_90: int = Field(ge=0, le=50, description="Number of loans that reached 90+ DPD")
    region_risk: Literal[0, 1, 2] = Field(description="Regional risk segment (0 = lowest)")
    credit_utilization: float | None = Field(default=None, ge=0, le=1,
                                             description="Credit utilisation ratio; imputed if missing")
    # Optional: not used by the current model, but accepted so the preprocessor sees the full record
    # (a future model can use them without an API change). Missing values are imputed.
    age: float | None = Field(default=None, ge=18, le=120, description="Applicant age in years (99 = unknown)")
    employment_years: float | None = Field(default=None, ge=0, le=80, description="Years in employment")
    num_loans: int | None = Field(default=None, ge=0, le=100, description="Number of loans")
    interest_rate: float | None = Field(default=None, ge=0, le=1, description="Offered interest rate (0.12 = 12%)")
    client_history_in_bank: int | None = Field(default=None, ge=0, le=1200,
                                               description="Relationship length with the bank (months)")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applications: list[LoanApplication] = Field(min_length=1)


class ReasonCode(BaseModel):
    code: str
    description: str
    impact: float = Field(description="Log-odds increase vs. an average applicant")


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    application_id: str | None
    probability_of_default: float = Field(ge=0, le=1)
    score: int = Field(description="Higher = lower risk (600 at 50:1 odds, PDO 20)")
    risk_grade: str
    reason_codes: list[ReasonCode] = Field(description="Main factors raising the PD, strongest first")
    warnings: list[str] = Field(description="Inputs adjusted during preprocessing")
    model_version: str


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_version: str
    model_type: str
    artifact_sha256: str
    features: list[str] = Field(description="Application fields the model uses")
    model_inputs: list[str] = Field(description="Preprocessed columns the model reads (region one-hot encoded)")
    master_scale: list[dict]
    validation: dict


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: Literal["ok", "unavailable"]
    model_version: str | None = None
