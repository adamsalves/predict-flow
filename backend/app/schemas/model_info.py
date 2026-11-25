"""Response DTO for ``GET /api/v1/model/info`` (issue #8).

Exposes what operators and the future UI need to audit the active model —
version, trusted digests, contract, operating point and headline historical
metrics — without leaking local filesystem layout or environment details.
Metrics are historical evaluations of the Telco label; the target
limitation text travels in the payload so no client can present the score
as a prospective churn probability (ADR-0009, ADR-0011).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


# See app/schemas/manifest.py for the justified explicit-any ignores.
class ModelInfoFeature(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    required: bool


class ModelInfoMetric(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    value: float
    ci95: tuple[float, float]


class ModelInfoSplitMetrics(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    roc_auc: ModelInfoMetric
    pr_auc: ModelInfoMetric
    brier: ModelInfoMetric
    base_rate: float


class ModelInfo(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    model_version: str
    trained_at: str
    git_commit: str
    artifact_sha256: str
    manifest_sha256: str
    published_at: str
    trusted_source: str
    dataset_name: str
    dataset_license: str
    dataset_sha256: str
    target_label: str
    target_limitations: str
    feature_schema_version: str
    feature_schema_sha256: str
    features: list[ModelInfoFeature]
    model_type: str
    threshold_value: float
    threshold_rule: str
    calibration_method: str
    explanation_method: str
    explanation_unit: str
    validation: ModelInfoSplitMetrics
    test_final: ModelInfoSplitMetrics
