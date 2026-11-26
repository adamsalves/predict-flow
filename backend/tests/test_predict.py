"""POST /api/v1/predict contract (issue #9).

HTTP-level behavior against the real published model and a throwaway
database: response shape, strict validation, historical labeling of demo
scores (ADR-0009), and the persisted audit row.
"""

import json
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.main import app
from app.models import ChurnPrediction, Customer
from app.schemas.manifest import ModelManifest
from tests.conftest import SAMPLE_INPUT


def _key() -> str:
    return f"test-{uuid.uuid4()}"


def _predict(
    client: TestClient, payload: dict[str, object], key: str
) -> httpx.Response:
    resp: httpx.Response = client.post(
        "/api/v1/predict", json=payload, headers={"Idempotency-Key": key}
    )
    return resp


def _manifest(published_models_dir: Path) -> ModelManifest:
    return ModelManifest.model_validate(
        json.loads(
            (published_models_dir / "v1" / "model_meta.json").read_text(
                encoding="utf-8"
            )
        )
    )


def _row_by_key(url: str, key: str) -> ChurnPrediction | None:
    engine = create_engine(url)
    try:
        with Session(engine) as session:
            return session.scalar(
                select(ChurnPrediction).where(
                    ChurnPrediction.idempotency_key == key
                )
            )
    finally:
        engine.dispose()


def test_valid_sandbox_prediction_returns_score_and_factors(
    predict_client: TestClient, published_models_dir: Path, migrated_db_url: str
) -> None:
    manifest = _manifest(published_models_dir)
    key = _key()
    resp = _predict(predict_client, dict(SAMPLE_INPUT), key)
    assert resp.status_code == 201
    body = resp.json()
    assert body["model_version"] == "v1"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["decision_threshold"] == manifest.model.threshold.value
    assert body["risk_level"] in ("high", "low")
    assert "historical" in body["target_limitations"].lower()
    assert "historical" in body["scope_note"].lower()
    assert body["feature_as_of"] is None  # Telco has no as-of data (ADR-0009)
    explanation = body["explanation"]
    assert explanation["method"] == manifest.explanation.method
    assert explanation["unit"] == manifest.explanation.unit
    assert explanation["model_version"] == "v1"
    assert explanation["top_k"] == manifest.explanation.top_k_default
    assert len(explanation["factors"]) == explanation["top_k"]
    for factor in explanation["factors"]:
        assert set(factor) == {"feature", "value", "contribution"}
        assert isinstance(factor["contribution"], float)
    assert "not causes" in explanation["interpretation_note"]

    row = _row_by_key(migrated_db_url, key)
    assert row is not None
    assert row.data_origin == "prospective"
    assert row.feature_as_of is None
    assert row.feature_snapshot == SAMPLE_INPUT
    assert row.factors["method"] == explanation["method"]
    assert row.model_version == "v1"
    assert row.customer_id is None
    assert row.is_current is False  # sandbox never joins the projection


def test_customer_prediction_is_current_and_registers_identity(
    predict_client: TestClient, migrated_db_url: str
) -> None:
    key = _key()
    external_id = f"cust-{_key()}"
    resp = _predict(
        predict_client, {**SAMPLE_INPUT, "customer_id_ext": external_id}, key
    )
    assert resp.status_code == 201
    row = _row_by_key(migrated_db_url, key)
    assert row is not None
    assert row.is_current is True
    engine = create_engine(migrated_db_url)
    try:
        with Session(engine) as session:
            customer = session.scalar(
                select(Customer).where(
                    Customer.customer_id_ext == external_id
                )
            )
    finally:
        engine.dispose()
    assert customer is not None
    assert row.customer_id == customer.id


def test_missing_feature_is_422(predict_client: TestClient) -> None:
    payload = dict(SAMPLE_INPUT)
    payload.pop("tenure")
    resp = _predict(predict_client, payload, _key())
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert "input" not in resp.text  # sanitized envelope (#40)


def test_wrong_type_is_422_not_coerced(predict_client: TestClient) -> None:
    resp = _predict(
        predict_client, {**SAMPLE_INPUT, "tenure": "12"}, _key()
    )
    assert resp.status_code == 422
    assert "tenure" in resp.text


def test_unknown_feature_is_422(predict_client: TestClient) -> None:
    resp = _predict(
        predict_client, {**SAMPLE_INPUT, "churn": 1}, _key()
    )
    assert resp.status_code == 422


def test_missing_idempotency_key_is_422(predict_client: TestClient) -> None:
    resp = predict_client.post("/api/v1/predict", json=dict(SAMPLE_INPUT))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_blank_idempotency_key_is_422(predict_client: TestClient) -> None:
    resp = _predict(predict_client, dict(SAMPLE_INPUT), "   ")
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [("MonthlyCharges", "89.5"), ("tenure", True), ("TotalCharges", None)],
)
def test_non_numeric_float_contract_is_422(
    predict_client: TestClient, field: str, value: object
) -> None:
    resp = _predict(predict_client, {**SAMPLE_INPUT, field: value}, _key())
    assert resp.status_code == 422


def test_predict_503_when_model_not_loaded(predict_client: TestClient) -> None:
    app.state.loaded_model = None
    resp = _predict(predict_client, dict(SAMPLE_INPUT), _key())
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "service_unavailable"


def test_openapi_documents_predict_contract(
    predict_client: TestClient,
) -> None:
    spec = predict_client.get("/api/openapi.json").json()
    operation = spec["paths"]["/api/v1/predict"]["post"]
    documented = set(operation["responses"])
    assert {"200", "201", "409", "422", "429", "500", "503"} <= documented
    ok = operation["responses"]["201"]["content"]["application/json"]["schema"]
    assert ok["$ref"].endswith("/PredictResponse")
    error = operation["responses"]["409"]["content"]["application/json"]["schema"]
    assert error["$ref"].endswith("/ErrorResponse")
    assert "PredictResponse" in spec["components"]["schemas"]


def test_app_state_is_clean_after_lifespan(predict_client: TestClient) -> None:
    assert app.state.loaded_model is not None
    assert app.state.predict_semaphore is not None
