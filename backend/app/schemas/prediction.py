"""Prediction request/response contracts (issue #9).

The request schema is static and mirrors the serving feature contract
(ADR-0013): the same names, types and optionality compiled into
``app/schemas/features.py``, with parity enforced by tests. Types are
validated strictly — a JSON string where the contract says ``int`` is a
422, not a silent coercion. ``customer_id_ext`` is the only optional
input: its absence marks an anonymous sandbox prediction.

The response carries the persisted result, reconstructed identically for
idempotent replays (ADR-0014): probability, provisional risk level,
the stable explanation DTO (ADR-0018) and the explicit temporal limits
of the demo target (ADR-0009).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, Strict

from app.schemas.explanation import LocalExplanation

# Scores produced by the demo API are historical/simulated: the Telco
# dataset has no observed prospective horizon (ADR-0009), so no response
# may read as a live "churn in N days" claim.
DEMO_SCOPE_NOTE = (
    "Demo score on historical/simulated data. The underlying dataset has "
    "no observed prospective horizon: this is not a live churn forecast "
    "and must not be presented as one."
)


def _numeric_only(value: object) -> object:
    """Float contract: JSON numbers only — never strings or booleans."""
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError("expected a JSON number")
    return value


# A JSON int where the contract says float is lossless; strings and
# booleans that would coerce silently are rejected above instead.
FloatFeature = Annotated[float, BeforeValidator(_numeric_only)]


class PredictRequest(BaseModel):  # type: ignore[explicit-any]  # inherited BaseModel.__init__(**data: Any); same justification as ExplanationFactor
    """One contract-shaped input row plus optional customer identity."""

    model_config = ConfigDict(extra="forbid")

    gender: Annotated[str, Strict()]
    SeniorCitizen: Annotated[int, Strict()]
    Partner: Annotated[str, Strict()]
    Dependents: Annotated[str, Strict()]
    tenure: Annotated[int, Strict()]
    PhoneService: Annotated[str, Strict()]
    MultipleLines: Annotated[str, Strict()]
    InternetService: Annotated[str, Strict()]
    OnlineSecurity: Annotated[str, Strict()]
    OnlineBackup: Annotated[str, Strict()]
    DeviceProtection: Annotated[str, Strict()]
    TechSupport: Annotated[str, Strict()]
    StreamingTV: Annotated[str, Strict()]
    StreamingMovies: Annotated[str, Strict()]
    Contract: Annotated[str, Strict()]
    PaperlessBilling: Annotated[str, Strict()]
    PaymentMethod: Annotated[str, Strict()]
    MonthlyCharges: FloatFeature
    TotalCharges: FloatFeature
    customer_id_ext: str | None = Field(
        default=None,
        description="Optional external customer id; absent = sandbox",
    )

    def features(self) -> dict[str, str | int | float | None]:
        """The validated input row as a plain dict (contract feature set)."""
        data = self.model_dump()
        data.pop("customer_id_ext")
        return data


class PredictResponse(BaseModel):  # type: ignore[explicit-any]  # see PredictRequest
    """The persisted prediction result, stable across idempotent replays."""

    prediction_id: uuid.UUID
    model_version: str
    score_at: datetime
    churn_probability: float = Field(ge=0.0, le=1.0)
    decision_threshold: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["high", "low"]
    target_label: str
    target_limitations: str
    # The Telco dataset carries no as-of timestamps: the absence stays
    # explicit in the contract instead of being faked by the API (ADR-0009).
    feature_as_of: None = None
    scope_note: str = DEMO_SCOPE_NOTE
    explanation: LocalExplanation
