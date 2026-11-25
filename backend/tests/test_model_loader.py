"""ModelLoader gates: provenance, contracts, environment, rollback (issue #8).

Every rejection case also proves ``joblib.load`` was never reached — the
artifact bytes are only unpickled after all gates pass (ADR-0017). The
published versions come from real ``training.train`` runs (see conftest),
so the registry under test is the one training actually writes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import joblib
import pytest

from app.services.model_loader import (
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    REGISTRY_FILENAME,
    ArtifactMissingError,
    CompatibilityError,
    ProvenanceError,
    load_active_model,
)
from tests.conftest import stage_versions
from training.publication import publish

ManifestMutator = Callable[[dict[str, object]], None]


def _registry_records(models_dir: Path) -> dict[str, dict[str, str]]:
    data = json.loads((models_dir / REGISTRY_FILENAME).read_text(encoding="utf-8"))
    return {r["model_version"]: r for r in data["publications"]}


def _edit_manifest(version_dir: Path, mutate: ManifestMutator) -> None:
    manifest_file = version_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_file.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _section(manifest: dict[str, object], key: str) -> dict[str, object]:
    value = manifest[key]
    assert isinstance(value, dict)
    return value


def _republish(version_dir: Path, registry_path: Path) -> None:
    """Re-anchor an edited version in a fresh trusted registry.

    Simulates a legitimately published but incompatible artifact, so the
    contract/environment gates are exercised independently of the digest
    gate (which the edit would otherwise trip first).
    """
    publish(version_dir, registry_path, git_commit="0" * 40)


def _forbid_joblib_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def _sentinel(path: object) -> object:
        raise AssertionError(
            "joblib.load must not run for artifacts rejected by the gates"
        )

    monkeypatch.setattr("app.services.model_loader.joblib_load", _sentinel)


def _staged_and_republished(
    published: Path, tmp_path: Path, mutate: ManifestMutator
) -> tuple[Path, Path]:
    models = stage_versions(published, tmp_path / "models", ("v1",))
    version_dir = models / "v1"
    _edit_manifest(version_dir, mutate)
    registry = tmp_path / "trusted_registry.json"
    _republish(version_dir, registry)
    return models, registry


def test_loads_published_version_and_predicts(
    published_models_dir: Path, sample_input: dict[str, object]
) -> None:
    loaded = load_active_model(model_path=published_models_dir, model_version="v1")
    record = _registry_records(published_models_dir)["v1"]
    assert loaded.model_version == "v1"
    assert loaded.publication.artifact_sha256 == record["artifact_sha256"]
    assert loaded.publication.manifest_sha256 == record["manifest_sha256"]
    assert loaded.trusted_source == REGISTRY_FILENAME
    proba = loaded.predict_proba(sample_input)
    assert 0.0 <= proba <= 1.0
    info = loaded.to_info()
    assert info.model_version == "v1"
    assert info.artifact_sha256 == record["artifact_sha256"]
    assert len(info.features) == 19


def test_predict_requires_every_contract_feature(
    published_models_dir: Path, sample_input: dict[str, object]
) -> None:
    loaded = load_active_model(model_path=published_models_dir, model_version="v1")
    del sample_input["tenure"]
    with pytest.raises(ValueError, match="missing contract features"):
        loaded.predict_proba(sample_input)


def test_missing_version_directory_fails(published_models_dir: Path) -> None:
    with pytest.raises(ArtifactMissingError, match="version directory not found"):
        load_active_model(model_path=published_models_dir, model_version="v9")


def test_missing_artifact_or_manifest_fails(
    published_models_dir: Path, tmp_path: Path
) -> None:
    models = stage_versions(published_models_dir, tmp_path / "models", ("v1",))
    (models / "v1" / MODEL_FILENAME).unlink()
    with pytest.raises(ArtifactMissingError, match="missing"):
        load_active_model(model_path=models, model_version="v1")


def test_missing_registry_fails(published_models_dir: Path, tmp_path: Path) -> None:
    models = stage_versions(
        published_models_dir, tmp_path / "models", ("v1",), with_registry=False
    )
    with pytest.raises(ArtifactMissingError, match="registry not found"):
        load_active_model(model_path=models, model_version="v1")


def test_unregistered_version_is_not_trusted(
    published_models_dir: Path, tmp_path: Path
) -> None:
    models = stage_versions(published_models_dir, tmp_path / "models", ("v1",))
    (models / "v1").rename(models / "v9")
    with pytest.raises(ProvenanceError, match="not present in the trusted registry"):
        load_active_model(model_path=models, model_version="v9")


def test_tampered_artifact_rejected_before_load(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = stage_versions(published_models_dir, tmp_path / "models", ("v1",))
    model_file = models / "v1" / MODEL_FILENAME
    content = bytearray(model_file.read_bytes())
    content[0] ^= 0xFF
    model_file.write_bytes(bytes(content))
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(ProvenanceError, match="diverge from the trusted"):
        load_active_model(model_path=models, model_version="v1")


def test_tampered_manifest_rejected_before_load(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = stage_versions(published_models_dir, tmp_path / "models", ("v1",))

    def _lower_threshold(manifest: dict[str, object]) -> None:
        _section(manifest, "model")["threshold"] = {"value": 0.01, "rule": "forged"}

    _edit_manifest(models / "v1", _lower_threshold)
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(ProvenanceError, match="diverge from the trusted"):
        load_active_model(model_path=models, model_version="v1")


def test_joint_artifact_manifest_substitution_rejected(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2's self-consistent artifact+manifest pair planted under v1's name.

    Both files agree with each other, so only the trusted registry — which
    still holds v1's original digests — can reject the swap (ADR-0017).
    """
    models = stage_versions(published_models_dir, tmp_path / "models", ("v1", "v2"))
    for filename in (MODEL_FILENAME, MANIFEST_FILENAME):
        (models / "v2" / filename).replace(models / "v1" / filename)
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(ProvenanceError, match="possible joint substitution"):
        load_active_model(model_path=models, model_version="v1")


def test_feature_schema_version_divergence_rejected(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _bump_version(manifest: dict[str, object]) -> None:
        _section(manifest, "features")["feature_schema_version"] = "999"

    models, registry = _staged_and_republished(
        published_models_dir, tmp_path, _bump_version
    )
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(CompatibilityError, match="feature schema version"):
        load_active_model(model_path=models, model_version="v1", registry_path=registry)


def test_feature_schema_digest_divergence_rejected(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _break_digest(manifest: dict[str, object]) -> None:
        _section(manifest, "features")["feature_schema_sha256"] = "0" * 64

    models, registry = _staged_and_republished(
        published_models_dir, tmp_path, _break_digest
    )
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(CompatibilityError, match="feature schema digest"):
        load_active_model(model_path=models, model_version="v1", registry_path=registry)


def test_feature_items_divergence_rejected(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _reverse_items(manifest: dict[str, object]) -> None:
        features = _section(manifest, "features")
        items = features["items"]
        assert isinstance(items, list)
        features["items"] = list(reversed(items))

    models, registry = _staged_and_republished(
        published_models_dir, tmp_path, _reverse_items
    )
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(CompatibilityError, match="feature list diverges"):
        load_active_model(model_path=models, model_version="v1", registry_path=registry)


def test_factors_schema_divergence_rejected(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _break_factors(manifest: dict[str, object]) -> None:
        _section(manifest, "factors_schema")["sha256"] = "0" * 64

    models, registry = _staged_and_republished(
        published_models_dir, tmp_path, _break_factors
    )
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(CompatibilityError, match="factors schema digest"):
        load_active_model(model_path=models, model_version="v1", registry_path=registry)


def test_unsupported_manifest_schema_rejected(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _downgrade(manifest: dict[str, object]) -> None:
        manifest["schema_version"] = "1"

    models, registry = _staged_and_republished(
        published_models_dir, tmp_path, _downgrade
    )
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(CompatibilityError, match="unsupported manifest"):
        load_active_model(model_path=models, model_version="v1", registry_path=registry)


def test_manifest_version_mismatch_rejected(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _rename(manifest: dict[str, object]) -> None:
        manifest["model_version"] = "v2"

    models, registry = _staged_and_republished(published_models_dir, tmp_path, _rename)
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(CompatibilityError, match="does not match the activated"):
        load_active_model(model_path=models, model_version="v1", registry_path=registry)


def test_manifest_artifact_claim_divergence_rejected(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _forge_claim(manifest: dict[str, object]) -> None:
        _section(manifest, "model")["artifact_sha256"] = "f" * 64

    models, registry = _staged_and_republished(
        published_models_dir, tmp_path, _forge_claim
    )
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(CompatibilityError, match="artifact_sha256 diverges"):
        load_active_model(model_path=models, model_version="v1", registry_path=registry)


def test_scikit_learn_version_divergence_rejected(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _downgrade_sklearn(manifest: dict[str, object]) -> None:
        _section(manifest, "environment")["scikit-learn"] = "0.0.0"

    models, registry = _staged_and_republished(
        published_models_dir, tmp_path, _downgrade_sklearn
    )
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(CompatibilityError, match="scikit-learn"):
        load_active_model(model_path=models, model_version="v1", registry_path=registry)


def test_numpy_major_divergence_rejected(
    published_models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _break_numpy(manifest: dict[str, object]) -> None:
        _section(manifest, "environment")["numpy"] = "1.26.4"

    models, registry = _staged_and_republished(
        published_models_dir, tmp_path, _break_numpy
    )
    _forbid_joblib_load(monkeypatch)
    with pytest.raises(CompatibilityError, match="numpy"):
        load_active_model(model_path=models, model_version="v1", registry_path=registry)


def test_numpy_same_major_accepted(
    published_models_dir: Path, tmp_path: Path
) -> None:
    def _other_minor(manifest: dict[str, object]) -> None:
        _section(manifest, "environment")["numpy"] = "2.0.0"

    models, registry = _staged_and_republished(
        published_models_dir, tmp_path, _other_minor
    )
    loaded = load_active_model(
        model_path=models, model_version="v1", registry_path=registry
    )
    assert loaded.model_version == "v1"


def test_non_scorer_artifact_rejected_after_gates(
    published_models_dir: Path, tmp_path: Path
) -> None:
    """A published non-model payload passes digests but fails the scorer gate."""
    models = stage_versions(published_models_dir, tmp_path / "models", ("v1",))
    version_dir = models / "v1"
    joblib.dump({"not": "a scorer"}, version_dir / MODEL_FILENAME)
    artifact_sha256 = hashlib.sha256(
        (version_dir / MODEL_FILENAME).read_bytes()
    ).hexdigest()

    def _align_claim(manifest: dict[str, object]) -> None:
        _section(manifest, "model")["artifact_sha256"] = artifact_sha256

    _edit_manifest(version_dir, _align_claim)
    registry = tmp_path / "trusted_registry.json"
    _republish(version_dir, registry)
    with pytest.raises(CompatibilityError, match="not a probabilistic scorer"):
        load_active_model(model_path=models, model_version="v1", registry_path=registry)


def test_rollback_between_compatible_versions(
    published_models_dir: Path, sample_input: dict[str, object]
) -> None:
    """Activation and rollback are config + restart between compatible
    versions (ADR-0003): both load through the same gates and the digests
    follow the active version."""
    activated = load_active_model(model_path=published_models_dir, model_version="v2")
    assert activated.model_version == "v2"

    rolled_back = load_active_model(model_path=published_models_dir, model_version="v1")
    assert rolled_back.model_version == "v1"
    assert (
        rolled_back.publication.artifact_sha256
        != activated.publication.artifact_sha256
    )
    assert 0.0 <= rolled_back.predict_proba(sample_input) <= 1.0
