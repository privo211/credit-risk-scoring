"""
Pydantic schemas for the Credit Risk Scoring API.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RiskBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Applicant(BaseModel):
    model_config = {"extra": "forbid"}

    checking_status: Optional[str] = Field(
        default=None,
        description="Status of existing checking account (A11-A14)",
    )
    duration: Optional[float] = Field(
        default=None, ge=0, description="Duration in months"
    )
    credit_history: Optional[str] = Field(
        default=None, description="Credit history (A30-A34)"
    )
    purpose: Optional[str] = Field(
        default=None, description="Purpose of loan (A40-A410)"
    )
    credit_amount: Optional[float] = Field(
        default=None, ge=0, description="Credit amount in DM"
    )
    savings: Optional[str] = Field(
        default=None, description="Savings account/bonds (A61-A65)"
    )
    employment: Optional[str] = Field(
        default=None, description="Present employment since (A71-A75)"
    )
    installment_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )
    personal_status: Optional[str] = Field(
        default=None, description="Personal status and sex (A91-A95)"
    )
    guarantors: Optional[str] = Field(
        default=None,
        description="Other debtors/guarantors (A101-A103)",
    )
    residence_since: Optional[float] = Field(
        default=None, ge=0, description="Present residence since (years)"
    )
    property: Optional[str] = Field(
        default=None, description="Property (A121-A124)"
    )
    age: Optional[float] = Field(
        default=None, ge=0, le=150,
    )
    other_plans: Optional[str] = Field(
        default=None, description="Other installment plans (A141-A145)"
    )
    housing: Optional[str] = Field(
        default=None, description="Housing (A151-A153)"
    )
    num_credits: Optional[float] = Field(
        default=None,
        ge=0,
    )
    job: Optional[str] = Field(
        default=None, description="Job category (A171-A174)"
    )
    people_maintenance: Optional[float] = Field(
        default=None,
        ge=0,
        description="Number of people liable for maintenance",
    )
    telephone: Optional[str] = Field(
        default=None, description="Telephone (A191-A192)"
    )
    foreign_worker: Optional[str] = Field(
        default=None, description="Foreign worker (A201-A202)"
    )


class PredictionResult(BaseModel):
    probability: float = Field(
        ..., ge=0, le=1, description="Predicted probability of default"
    )
    risk_band: RiskBand = Field(
        ...,
    )
    model_version: str = Field(
        ...,
    )


class BatchPredictionRequest(BaseModel):
    applicants: list[Applicant] = Field(
        ..., min_length=1,
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResult] = Field(
        ...,
    )
    count: int = Field(
        ...,
    )


class HealthResponse(BaseModel):
    status: str = Field(
        ...,
    )
    model_loaded: bool = Field(
        ...,
    )
    version: str = Field(
        ...,
    )
