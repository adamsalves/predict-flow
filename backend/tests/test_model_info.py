"""Lifespan activation and ``GET /api/v1/model/info`` (issue #8).

Activation/rollback rehearsal at the API level: switching versions is a
configuration change plus a process restart — modeled here by a fresh
``TestClient`` context per configuration (ADR-0003).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services.model_loader import (
    REGISTRY_FILENAME,
    LoadedModel,
    ProvenanceError,
    load_active_model,
)
from tests.conftest import stage_versions


@pytest.fixture
def model_env(
    monkeypatch: pytest.MonkeyPatch, published_models_dir: Path
) -> Iterator[None]:
    monkeypatch.setenv("MODEL_PATH", str(published_models_dir))
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.delenv("MODEL_REGISTRY_PATH", raising=False)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def _registry_records(models_dir: Path) -> dict[str, dict[str, str]]:
    data = json.loads((models_dir / REGISTRY_FILENAME).read_text(encoding="utf-8"))
    return {r["model_version"]: r for r in data["publications"]}


def test_model_info_returns_active_version(
    model_env: None, published_models_dir: Path
) -> None:
    with TestClient(app) as client:
        resp = client.get("/api/v1/model/info")
    assert resp.status_code == 200
    body = resp.json()
    record = _registry_records(published_models_dir)["v1"]
    assert body["model_version"] == "v1"
    assert body["artifact_sha256"] == record["artifact_sha256"]
    assert body["manifest_sha256"] == record["manifest_sha256"]
    assert body["trusted_source"] == REGISTRY_FILENAME
    assert body["published_at"]
    assert body["git_commit"]
    assert body["dataset_name"] == "synthetic mini dataset"
    assert "historical" in body["target_limitations"].lower()
    assert len(body["features"]) == 19
    assert body["feature_schema_version"] == "1"
    assert 0.0 < body["threshold_value"] < 1.0
    assert body["calibration_method"] in {"none", "sigmoid", "isotonic"}
    assert body["explanation_method"] == "linear"
    assert 0.0 <= body["validation"]["roc_auc"]["value"] <= 1.0
    assert 0.0 <= body["test_final"]["pr_auc"]["value"] <= 1.0
    # The response must not leak the local filesystem layout.
    assert str(published_models_dir) not in resp.text


def test_model_info_503_when_model_not_loaded() -> None:
    app.state.loaded_model = None
    resp = TestClient(app).get("/api/v1/model/info")
    assert resp.status_code == 503


def test_model_loaded_once_during_lifespan(
    model_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def spy(
        *,
        model_path: Path,
        model_version: str,
        registry_path: Path | None = None,
    ) -> LoadedModel:
        nonlocal calls
        calls += 1
        return load_active_model(
            model_path=model_path,
            model_version=model_version,
            registry_path=registry_path,
        )

    monkeypatch.setattr("app.main.load_active_model", spy)
    with TestClient(app) as client:
        client.get("/api/v1/health/live")
        client.get("/api/v1/model/info")
        client.get("/api/v1/model/info")
    assert calls == 1


def test_activation_and_rollback_via_config_and_restart(
    model_env: None, monkeypatch: pytest.MonkeyPatch, published_models_dir: Path
) -> None:
    records = _registry_records(published_models_dir)
    with TestClient(app) as client:
        body = client.get("/api/v1/model/info").json()
        assert body["model_version"] == "v1"
        assert body["artifact_sha256"] == records["v1"]["artifact_sha256"]

    # Activate v2: configuration change + restart.
    monkeypatch.setenv("MODEL_VERSION", "v2")
    get_settings.cache_clear()
    with TestClient(app) as client:
        body = client.get("/api/v1/model/info").json()
        assert body["model_version"] == "v2"
        assert body["artifact_sha256"] == records["v2"]["artifact_sha256"]

    # Roll back to v1: point the configuration back and restart again.
    monkeypatch.setenv("MODEL_VERSION", "v1")
    get_settings.cache_clear()
    with TestClient(app) as client:
        body = client.get("/api/v1/model/info").json()
        assert body["model_version"] == "v1"
        assert body["artifact_sha256"] == records["v1"]["artifact_sha256"]


def test_startup_aborts_when_provenance_fails(
    model_env: None,
    monkeypatch: pytest.MonkeyPatch,
    published_models_dir: Path,
    tmp_path: Path,
) -> None:
    models = stage_versions(published_models_dir, tmp_path / "models", ("v1",))
    model_file = models / "v1" / "model.joblib"
    content = bytearray(model_file.read_bytes())
    content[0] ^= 0xFF
    model_file.write_bytes(bytes(content))
    monkeypatch.setenv("MODEL_PATH", str(models))
    get_settings.cache_clear()
    with pytest.raises(ProvenanceError, match="diverge from the trusted"):
        with TestClient(app):
            pass
