"""Stable local-explanation DTO shared by every model adapter (ADR-0018).

The frontend consumes a single contract regardless of the active
algorithm: LR and TreeSHAP adapters both produce ``LocalExplanation``.
The unit and meaning of ``contribution`` are explicit and must never be
compared or summed across methods/models with different units. The schema
is versioned; ``factors_schema_digest`` anchors the contract in the model
manifest (ADR-0002), like the feature contract digest (ADR-0013).
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

FACTORS_SCHEMA_VERSION = "1"

# ADR-0011: explanations are predictive, never causal — declared in the
# payload itself so no client can present them otherwise.
PREDICTIVE_NOT_CAUSAL_NOTE = (
    "Contributions explain this model's predictive output for this input. "
    "They are not causes of churn and do not estimate the effect of any "
    "retention action."
)
CALIBRATED_NOTE = (
    " Contributions are computed on the uncalibrated base estimator; "
    "calibration is a monotone remap of the final score."
)

METHOD_LINEAR = "linear"
UNIT_LOG_ODDS = "log_odds"
METHOD_TREE_SHAP = "tree_shap"
UNIT_PROBABILITY = "probability"


# mypy's disallow_any_explicit flags the inherited BaseModel.__init__(**data: Any)
# at the class definition; the Any lives in the library signature, so no local
# typing change can silence it. Justification recorded per project convention.
class ExplanationFactor(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    feature: str = Field(description="Contract feature name (ADR-0013)")
    value: str | int | float | None = Field(
        description="Raw input value for this feature, before preprocessing"
    )
    contribution: float = Field(
        description="Signed contribution in the explanation's declared unit"
    )


class LocalExplanation(BaseModel):  # type: ignore[explicit-any]  # see ExplanationFactor
    model_config = ConfigDict(frozen=True)

    schema_version: str = FACTORS_SCHEMA_VERSION
    method: str = Field(description="Explanation method, e.g. linear, tree_shap")
    unit: str = Field(description="Contribution unit, e.g. log_odds, probability")
    model_version: str
    base_value: float = Field(
        description="Expected model output before seeing this input"
    )
    output_value: float = Field(
        description="Model output for this input (base + all contributions)"
    )
    top_k: int
    factors: list[ExplanationFactor]
    interpretation_note: str = PREDICTIVE_NOT_CAUSAL_NOTE


def factors_schema_digest() -> str:
    """SHA-256 of the canonical explanation schema (order-insensitive keys)."""
    canonical = {
        "envelope": [
            {"name": "schema_version", "type": "str"},
            {"name": "method", "type": "str"},
            {"name": "unit", "type": "str"},
            {"name": "model_version", "type": "str"},
            {"name": "base_value", "type": "float"},
            {"name": "output_value", "type": "float"},
            {"name": "top_k", "type": "int"},
            {"name": "factors", "type": "list[ExplanationFactor]"},
            {"name": "interpretation_note", "type": "str"},
        ],
        "factor": [
            {"name": "feature", "type": "str"},
            {"name": "value", "type": "str | int | float | None"},
            {"name": "contribution", "type": "float"},
        ],
    }
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
