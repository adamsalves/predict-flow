"""Serving-side projection of the model manifest (schema_version "2").

Training owns the full manifest (ADR-0002); the API parses only the
sections it enforces or exposes — provenance, feature/factors contracts,
environment compatibility, operating point and headline metrics. Unknown
fields are ignored on purpose: the digest gates (ADR-0017) already bind
the exact bytes of the document, and the projection keeps serving decoupled
from training-only sections (selection, splits, bootstrap methodology).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# mypy's disallow_any_explicit flags the inherited BaseModel.__init__(**data: Any)
# at every model definition; the Any lives in the library signature, so no local
# typing change can silence it. Justification recorded per project convention
# (same pattern as app/schemas/explanation.py).


class ManifestGit(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    commit: str


class ManifestDataset(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    license: str
    sha256: str
    n_rows: int


class ManifestTarget(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    label: str
    definition: str
    limitations: str


class ManifestFeatureItem(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    dtype: str
    required: bool


class ManifestFeatures(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    feature_schema_version: str
    feature_schema_sha256: str
    items: list[ManifestFeatureItem]


class ManifestFactorsSchema(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: str
    sha256: str


class ManifestThreshold(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    value: float
    rule: str


class ManifestCalibration(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    method: str


class ManifestModel(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str
    artifact_sha256: str
    threshold: ManifestThreshold
    calibration: ManifestCalibration


class ManifestExplanation(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    method: str
    unit: str
    top_k_default: int


class ManifestSplitMetrics(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    roc_auc: float
    roc_auc_ci95: tuple[float, float]
    pr_auc: float
    pr_auc_ci95: tuple[float, float]
    brier: float
    brier_ci95: tuple[float, float]
    base_rate: float


class ManifestMetrics(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    validation: ManifestSplitMetrics
    test_final: ManifestSplitMetrics


class ManifestEnvironment(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    python: str
    scikit_learn: str = Field(alias="scikit-learn")
    numpy: str
    joblib: str


class ModelManifest(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: str
    model_version: str
    trained_at: str
    git: ManifestGit
    dataset: ManifestDataset
    target: ManifestTarget
    features: ManifestFeatures
    factors_schema: ManifestFactorsSchema
    model: ManifestModel
    explanation: ManifestExplanation
    metrics: ManifestMetrics
    environment: ManifestEnvironment
