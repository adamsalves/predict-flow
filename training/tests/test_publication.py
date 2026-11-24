"""Publication registry: trusted digest source, immutability, substitution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.publication import (
    REGISTRY_FILENAME,
    PublicationError,
    publish,
    verify_publication,
)


def _make_version(root: Path, version: str, payload: bytes) -> Path:
    version_dir = root / version
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "model.joblib").write_bytes(payload)
    (version_dir / "model_meta.json").write_text(
        json.dumps({"model_version": version, "artifact": payload.hex()}),
        encoding="utf-8",
    )
    return version_dir


def test_publish_records_digests_outside_the_version_dir(tmp_path: Path) -> None:
    version_dir = _make_version(tmp_path, "v1", b"model-bytes")
    record = publish(version_dir, git_commit="abc123")

    registry_path = tmp_path / REGISTRY_FILENAME
    assert registry_path.is_file()
    assert record.model_version == "v1"
    data = json.loads(registry_path.read_text())
    assert data["publications"][0]["artifact_sha256"] == record.artifact_sha256
    assert data["publications"][0]["manifest_sha256"] == record.manifest_sha256
    assert record.git_commit == "abc123"


def test_verify_roundtrip_passes(tmp_path: Path) -> None:
    version_dir = _make_version(tmp_path, "v1", b"model-bytes")
    publish(version_dir, git_commit="abc123")
    record = verify_publication(version_dir)
    assert record.model_version == "v1"


def test_joint_artifact_manifest_substitution_is_rejected(tmp_path: Path) -> None:
    version_dir = _make_version(tmp_path, "v1", b"original")
    publish(version_dir, git_commit="abc123")

    # Attacker replaces BOTH files with a self-consistent pair: the adjacent
    # manifest still matches the artifact, but the trusted registry does not.
    _make_version(tmp_path, "v1", b"forged")
    with pytest.raises(PublicationError, match="diverge from the trusted"):
        verify_publication(version_dir)


def test_manifest_only_tamper_is_rejected(tmp_path: Path) -> None:
    version_dir = _make_version(tmp_path, "v1", b"original")
    publish(version_dir, git_commit="abc123")
    (version_dir / "model_meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PublicationError, match="diverge"):
        verify_publication(version_dir)


def test_republish_same_content_is_idempotent(tmp_path: Path) -> None:
    version_dir = _make_version(tmp_path, "v1", b"model-bytes")
    first = publish(version_dir, git_commit="abc123")
    second = publish(version_dir, git_commit="abc123")
    assert first.artifact_sha256 == second.artifact_sha256
    data = json.loads((tmp_path / REGISTRY_FILENAME).read_text())
    assert len(data["publications"]) == 1


def test_republish_different_content_is_refused(tmp_path: Path) -> None:
    version_dir = _make_version(tmp_path, "v1", b"original")
    publish(version_dir, git_commit="abc123")
    _make_version(tmp_path, "v1", b"changed")
    with pytest.raises(PublicationError, match="immutable"):
        publish(version_dir, git_commit="abc123")


def test_verify_without_registry_fails(tmp_path: Path) -> None:
    version_dir = _make_version(tmp_path, "v1", b"model-bytes")
    with pytest.raises(PublicationError, match="registry not found"):
        verify_publication(version_dir)


def test_verify_unknown_version_fails(tmp_path: Path) -> None:
    published = _make_version(tmp_path, "v1", b"model-bytes")
    publish(published, git_commit="abc123")
    other = _make_version(tmp_path, "v2", b"other")
    with pytest.raises(PublicationError, match="not present"):
        verify_publication(other)
