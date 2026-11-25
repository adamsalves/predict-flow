"""Model artifact loading behind provenance and contract gates (issue #8).

Single validation path shared by every consumer — the API lifespan today,
batch workers (#18) tomorrow — so no code path can load an artifact without
the full gate sequence (ADR-0003, ADR-0013, ADR-0017):

1. presence of the version directory, artifact and manifest;
2. provenance: artifact and manifest SHA-256 verified against the trusted
   publication registry (kept outside the version directory), before the
   manifest is even parsed — a joint artifact+manifest substitution with
   self-consistent hashes is rejected here;
3. manifest structure: supported schema version, version identity and
   agreement with the trusted artifact digest;
4. contracts: feature schema version/digest/items and factors schema
   digest must match the contracts compiled into this API build;
5. environment: scikit-learn must match the training pin exactly and
   numpy must share the same major version (the locks already enforce
   parity, ADR-0017 — these gates are the runtime tripwire for artifacts
   trained elsewhere; the Python minor version is deliberately not gated
   because the pickle protocol is stable across the supported range);
6. only then ``joblib.load``, which unpickles potentially unsafe content
   and therefore runs strictly after every gate.

The registry verification mirrors ``training/publication.py``: the serving
image does not ship the training package, so the registry JSON format is
the shared contract and parity is enforced by tests that publish with
training and load here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import ValidationError

from app.schemas.explanation import FACTORS_SCHEMA_VERSION, factors_schema_digest
from app.schemas.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    feature_schema_digest,
)
from app.schemas.manifest import ManifestSplitMetrics, ModelManifest
from app.schemas.model_info import (
    ModelInfo,
    ModelInfoFeature,
    ModelInfoMetric,
    ModelInfoSplitMetrics,
)

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "model_meta.json"
REGISTRY_FILENAME = "publication_registry.json"
REGISTRY_SCHEMA_VERSION = "1"
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({"2"})
_RECORD_FIELDS = (
    "model_version",
    "artifact_sha256",
    "manifest_sha256",
    "published_at",
    "git_commit",
)

# Bound at module level so tests can replace it with a sentinel and prove
# that rejected artifacts never reach joblib.load.
joblib_load = joblib.load


class ModelLoadError(Exception):
    """Base class for activation failures; all of them abort startup."""


class ArtifactMissingError(ModelLoadError):
    """Version directory, artifact, manifest or registry is absent."""


class ProvenanceError(ModelLoadError):
    """Digests diverge from the trusted registry or the version is unknown."""


class CompatibilityError(ModelLoadError):
    """Manifest, feature/factors contracts or environment are incompatible."""


@dataclass(frozen=True)
class TrustedPublication:
    """The trusted-registry record that anchors activation (ADR-0017)."""

    model_version: str
    artifact_sha256: str
    manifest_sha256: str
    published_at: str
    git_commit: str


@runtime_checkable
class ProbabilisticScorer(Protocol):
    """The serving contract of an activated artifact."""

    def predict_proba(self, X: pd.DataFrame) -> object: ...


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(registry_path: Path) -> list[dict[str, str]]:
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceError(
            f"publication registry is not valid JSON: {registry_path}"
        ) from exc
    if not isinstance(data, dict):
        raise ProvenanceError("publication registry must be a JSON object")
    if data.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ProvenanceError(
            f"unsupported registry schema_version: {data.get('schema_version')!r}"
        )
    publications = data.get("publications")
    if not isinstance(publications, list):
        raise ProvenanceError("registry has no publications list")
    records: list[dict[str, str]] = []
    for record in publications:
        if not isinstance(record, dict) or any(
            not isinstance(record.get(field), str) for field in _RECORD_FIELDS
        ):
            raise ProvenanceError(f"malformed registry record: {record!r}")
        records.append({field: str(record[field]) for field in _RECORD_FIELDS})
    return records


def _verify_publication(version_dir: Path, registry_path: Path) -> TrustedPublication:
    """Check the version's files against the trusted registry digests.

    Raises ProvenanceError when the registry or the version entry is
    missing, or when either digest diverges — including the case where
    artifact and manifest were substituted together with self-consistent
    hashes, which the adjacent manifest alone could never detect.
    """
    if not registry_path.is_file():
        raise ArtifactMissingError(
            f"publication registry not found: {registry_path}"
        )
    records = _load_records(registry_path)
    artifact_sha256 = _sha256_file(version_dir / MODEL_FILENAME)
    manifest_sha256 = _sha256_file(version_dir / MANIFEST_FILENAME)
    for record in records:
        if record["model_version"] != version_dir.name:
            continue
        if (
            record["artifact_sha256"] != artifact_sha256
            or record["manifest_sha256"] != manifest_sha256
        ):
            raise ProvenanceError(
                f"digests of {version_dir.name!r} diverge from the trusted "
                "registry; refusing activation (possible joint substitution)"
            )
        return TrustedPublication(**record)
    raise ProvenanceError(
        f"version {version_dir.name!r} is not present in the trusted registry"
    )


def _parse_manifest(manifest_file: Path, model_version: str) -> ModelManifest:
    try:
        raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompatibilityError(
            f"manifest is not valid JSON: {manifest_file}"
        ) from exc
    try:
        manifest = ModelManifest.model_validate(raw)
    except ValidationError as exc:
        raise CompatibilityError(
            f"manifest does not match the serving projection: {exc}"
        ) from exc
    if manifest.schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise CompatibilityError(
            f"unsupported manifest schema_version: {manifest.schema_version!r}"
        )
    if manifest.model_version != model_version:
        raise CompatibilityError(
            f"manifest model_version {manifest.model_version!r} does not "
            f"match the activated version {model_version!r}"
        )
    return manifest


def _check_contracts(manifest: ModelManifest) -> None:
    features = manifest.features
    if features.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise CompatibilityError(
            f"feature schema version {features.feature_schema_version!r} is "
            f"not servable by this API ({FEATURE_SCHEMA_VERSION!r})"
        )
    if features.feature_schema_sha256 != feature_schema_digest():
        raise CompatibilityError(
            "feature schema digest diverges from the contract served by "
            "this API; refusing activation (ADR-0013)"
        )
    served = [(spec.name, spec.dtype, spec.required) for spec in FEATURES]
    recorded = [(item.name, item.dtype, item.required) for item in features.items]
    if recorded != served:
        raise CompatibilityError(
            "manifest feature list diverges from the contract served by "
            "this API (ADR-0013)"
        )
    factors = manifest.factors_schema
    if factors.schema_version != FACTORS_SCHEMA_VERSION:
        raise CompatibilityError(
            f"factors schema version {factors.schema_version!r} is not "
            f"servable by this API ({FACTORS_SCHEMA_VERSION!r})"
        )
    if factors.sha256 != factors_schema_digest():
        raise CompatibilityError(
            "factors schema digest diverges from the contract served by "
            "this API (ADR-0018)"
        )


def _check_environment(manifest: ModelManifest) -> None:
    environment = manifest.environment
    if sklearn.__version__ != environment.scikit_learn:
        raise CompatibilityError(
            f"artifact was trained with scikit-learn "
            f"{environment.scikit_learn}, serving runs "
            f"{sklearn.__version__}; the pipeline must match exactly "
            "(ADR-0017)"
        )
    serving_numpy_major = np.__version__.split(".")[0]
    if environment.numpy.split(".")[0] != serving_numpy_major:
        raise CompatibilityError(
            f"artifact was trained with numpy {environment.numpy}, serving "
            f"runs {np.__version__}; major versions must match"
        )


def _split_metrics_info(metrics: ManifestSplitMetrics) -> ModelInfoSplitMetrics:
    return ModelInfoSplitMetrics(
        roc_auc=ModelInfoMetric(
            value=metrics.roc_auc, ci95=metrics.roc_auc_ci95
        ),
        pr_auc=ModelInfoMetric(value=metrics.pr_auc, ci95=metrics.pr_auc_ci95),
        brier=ModelInfoMetric(value=metrics.brier, ci95=metrics.brier_ci95),
        base_rate=metrics.base_rate,
    )


@dataclass(frozen=True)
class LoadedModel:
    """An activated artifact: scorer plus its verified provenance."""

    scorer: ProbabilisticScorer
    manifest: ModelManifest
    publication: TrustedPublication
    trusted_source: str

    @property
    def model_version(self) -> str:
        return self.manifest.model_version

    def predict_proba(self, features: Mapping[str, object]) -> float:
        """Score one contract-shaped input row.

        Request-level validation and coercion land with the inference
        endpoint (#9); here every contract feature must be present and the
        fitted Pipeline handles preprocessing internally.
        """
        missing = [name for name in FEATURE_NAMES if name not in features]
        if missing:
            raise ValueError(
                f"prediction input is missing contract features: {missing}"
            )
        frame = pd.DataFrame({name: [features[name]] for name in FEATURE_NAMES})
        proba = np.asarray(self.scorer.predict_proba(frame), dtype=np.float64)
        return float(proba[0, 1])

    def to_info(self) -> ModelInfo:
        manifest = self.manifest
        return ModelInfo(
            model_version=manifest.model_version,
            trained_at=manifest.trained_at,
            git_commit=manifest.git.commit,
            artifact_sha256=self.publication.artifact_sha256,
            manifest_sha256=self.publication.manifest_sha256,
            published_at=self.publication.published_at,
            trusted_source=self.trusted_source,
            dataset_name=manifest.dataset.name,
            dataset_license=manifest.dataset.license,
            dataset_sha256=manifest.dataset.sha256,
            target_label=manifest.target.label,
            target_limitations=manifest.target.limitations,
            feature_schema_version=manifest.features.feature_schema_version,
            feature_schema_sha256=manifest.features.feature_schema_sha256,
            features=[
                ModelInfoFeature(
                    name=item.name, dtype=item.dtype, required=item.required
                )
                for item in manifest.features.items
            ],
            model_type=manifest.model.type,
            threshold_value=manifest.model.threshold.value,
            threshold_rule=manifest.model.threshold.rule,
            calibration_method=manifest.model.calibration.method,
            explanation_method=manifest.explanation.method,
            explanation_unit=manifest.explanation.unit,
            validation=_split_metrics_info(manifest.metrics.validation),
            test_final=_split_metrics_info(manifest.metrics.test_final),
        )


def load_active_model(
    *,
    model_path: Path,
    model_version: str,
    registry_path: Path | None = None,
) -> LoadedModel:
    """Activate one published version, or raise and abort startup.

    The trusted digest source defaults to the publication registry at the
    root of the models volume (outside the version directory, ADR-0017);
    ``registry_path`` allows pointing at an approved copy elsewhere.
    """
    version_dir = model_path / model_version
    model_file = version_dir / MODEL_FILENAME
    manifest_file = version_dir / MANIFEST_FILENAME
    if not version_dir.is_dir():
        raise ArtifactMissingError(
            f"model version directory not found: {version_dir}"
        )
    missing = [
        path.name for path in (model_file, manifest_file) if not path.is_file()
    ]
    if missing:
        raise ArtifactMissingError(
            f"version directory {version_dir} is missing: {', '.join(missing)}"
        )

    registry = (
        registry_path if registry_path is not None else model_path / REGISTRY_FILENAME
    )
    publication = _verify_publication(version_dir, registry)
    manifest = _parse_manifest(manifest_file, model_version)
    _check_contracts(manifest)
    _check_environment(manifest)
    if manifest.model.artifact_sha256 != publication.artifact_sha256:
        raise CompatibilityError(
            "manifest artifact_sha256 diverges from the trusted registry "
            "record; refusing activation"
        )

    scorer = joblib_load(model_file)
    if not isinstance(scorer, ProbabilisticScorer):
        raise CompatibilityError(
            f"artifact is not a probabilistic scorer: {type(scorer).__name__}"
        )
    logger.info(
        "model %s activated (artifact sha256 %s, trusted source %s)",
        model_version,
        publication.artifact_sha256,
        registry,
    )
    return LoadedModel(
        scorer=scorer,
        manifest=manifest,
        publication=publication,
        trusted_source=registry.name,
    )
