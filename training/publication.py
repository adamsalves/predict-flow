"""Artifact publication registry: the trusted digest source (ADR-0017).

The manifest adjacent to the artifact is not the only anchor of trust:
publication appends the version's digests to a registry kept outside the
version directory, and activation (issue #8) requires the digest approved
by this trusted source before ``joblib.load``. A joint artifact+manifest
substitution is rejected because the registry digest no longer matches,
and a published version is immutable: republishing the same version with
different content is an error, while an identical republish is a no-op.

Concurrency note: publication is read-modify-write without locking — safe
for the single-process local demo (ADR-0010); batch workers (#12/#16) must
revisit this before publishing concurrently.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from training.dataset import sha256_file

MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "model_meta.json"
REGISTRY_FILENAME = "publication_registry.json"
REGISTRY_SCHEMA_VERSION = "1"
_RECORD_FIELDS = (
    "model_version",
    "artifact_sha256",
    "manifest_sha256",
    "published_at",
    "git_commit",
)


class PublicationError(Exception):
    """Raised when publication or verification against the registry fails."""


@dataclass(frozen=True)
class PublicationRecord:
    model_version: str
    artifact_sha256: str
    manifest_sha256: str
    published_at: str
    git_commit: str


def _default_registry_path(version_dir: Path) -> Path:
    return version_dir.parent / REGISTRY_FILENAME


def _load_registry(registry_path: Path) -> dict[str, object]:
    if not registry_path.is_file():
        return {"schema_version": REGISTRY_SCHEMA_VERSION, "publications": []}
    try:
        data: dict[str, object] = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PublicationError(f"registry is not valid JSON: {registry_path}") from exc
    if data.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise PublicationError(
            f"unsupported registry schema_version: {data.get('schema_version')!r}"
        )
    if not isinstance(data.get("publications"), list):
        raise PublicationError("registry has no publications list")
    publications = data["publications"]
    assert isinstance(publications, list)
    for record in publications:
        if not isinstance(record, dict) or any(
            not isinstance(record.get(field), str) for field in _RECORD_FIELDS
        ):
            raise PublicationError(f"malformed registry record: {record!r}")
    return data


def _write_registry(registry_path: Path, data: dict[str, object]) -> None:
    tmp = registry_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(registry_path)


def _records(data: dict[str, object]) -> list[dict[str, str]]:
    publications = data["publications"]
    assert isinstance(publications, list)
    return publications


def _digests(version_dir: Path) -> tuple[str, str]:
    model_path = version_dir / MODEL_FILENAME
    manifest_path = version_dir / MANIFEST_FILENAME
    if not model_path.is_file() or not manifest_path.is_file():
        raise PublicationError(
            f"version directory {version_dir} must contain "
            f"{MODEL_FILENAME} and {MANIFEST_FILENAME}"
        )
    return sha256_file(model_path), sha256_file(manifest_path)


def publish(
    version_dir: Path,
    registry_path: Path | None = None,
    *,
    git_commit: str,
) -> PublicationRecord:
    """Record the version's digests in the trusted registry.

    Idempotent for identical content; refuses to overwrite a published
    version whose digests differ (published versions are immutable).
    """
    registry = registry_path or _default_registry_path(version_dir)
    artifact_sha256, manifest_sha256 = _digests(version_dir)
    data = _load_registry(registry)

    for existing in _records(data):
        if existing["model_version"] != version_dir.name:
            continue
        if (
            existing["artifact_sha256"] == artifact_sha256
            and existing["manifest_sha256"] == manifest_sha256
        ):
            return PublicationRecord(**existing)
        raise PublicationError(
            f"version {version_dir.name!r} is already published with "
            "different digests; published versions are immutable"
        )

    record = PublicationRecord(
        model_version=version_dir.name,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        published_at=datetime.now(UTC).isoformat(),
        git_commit=git_commit,
    )
    _records(data).append(asdict(record))
    _write_registry(registry, data)
    return record


def verify_publication(
    version_dir: Path, registry_path: Path | None = None
) -> PublicationRecord:
    """Check the version's files against the trusted registry digests.

    Raises PublicationError when the registry or the version entry is
    missing, or when either digest diverges — including the case where
    artifact and manifest were substituted together with self-consistent
    hashes, which the adjacent manifest alone could never detect.
    """
    registry = registry_path or _default_registry_path(version_dir)
    if not registry.is_file():
        raise PublicationError(f"publication registry not found: {registry}")
    data = _load_registry(registry)
    artifact_sha256, manifest_sha256 = _digests(version_dir)
    for existing in _records(data):
        if existing["model_version"] != version_dir.name:
            continue
        if (
            existing["artifact_sha256"] != artifact_sha256
            or existing["manifest_sha256"] != manifest_sha256
        ):
            raise PublicationError(
                f"digests of {version_dir.name!r} diverge from the trusted "
                "registry; refusing activation (possible joint substitution)"
            )
        return PublicationRecord(**existing)
    raise PublicationError(
        f"version {version_dir.name!r} is not present in the trusted registry"
    )
