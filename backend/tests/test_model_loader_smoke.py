"""Real-artifact serving smoke: carregar -> predizer (issue #8, M2 gate).

CI runs this file after the training smoke, with ``PF_SMOKE_MODELS_DIR``
pointing at the directory where ``python -m training.train --version
ci-smoke`` published a real Telco-trained artifact. Everywhere else it
self-skips, like the Postgres-backed tests. The smoke exercises the whole
serving path end to end: environment settings -> lifespan -> registry,
contract and environment gates -> ``joblib.load`` -> ``predict_proba`` ->
``GET /api/v1/model/info``. It is deliberately self-contained (its own
sample input) so it can run as a standalone pytest invocation.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

MODELS_DIR = os.environ.get("PF_SMOKE_MODELS_DIR")
MODEL_VERSION = os.environ.get("PF_SMOKE_MODEL_VERSION", "ci-smoke")

pytestmark = pytest.mark.skipif(
    MODELS_DIR is None,
    reason="PF_SMOKE_MODELS_DIR não definido; smoke com artefato real pulado",
)

SAMPLE_INPUT: dict[str, object] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 1074.0,
}


@pytest.fixture
def smoke_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    assert MODELS_DIR is not None
    monkeypatch.setenv("MODEL_PATH", MODELS_DIR)
    monkeypatch.setenv("MODEL_VERSION", MODEL_VERSION)
    monkeypatch.delenv("MODEL_REGISTRY_PATH", raising=False)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def test_real_artifact_loads_and_predicts(smoke_env: None) -> None:
    with TestClient(app) as client:
        loaded = app.state.loaded_model
        assert loaded.model_version == MODEL_VERSION
        proba = loaded.predict_proba(SAMPLE_INPUT)
        assert 0.0 <= proba <= 1.0

        resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_version"] == MODEL_VERSION
        assert body["artifact_sha256"] == loaded.publication.artifact_sha256
        assert body["dataset_name"] == "Telco Customer Churn (IBM)"
