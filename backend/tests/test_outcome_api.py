"""PATCH /predictions/{id}/outcome: HTTP layer of the outcome API (issue #10).

A thin route over the idempotent service: retries replay byte-identical
state, domain failures map to the common envelope, and the DTO rejects
what the client may never choose — extra fields (the outcome origin is
derived from the prediction, D6), naive datetimes, blank idempotency keys.
"""

import uuid
from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import ChurnPrediction, Customer, DataOrigin, OutcomeIdempotency
from app.schemas.outcome import OutcomePatchRequest
from app.services.customers import register_customer
from app.services.predictions import record_prediction
from tests.conftest import SAMPLE_INPUT


def _key() -> str:
    return f"test-{uuid.uuid4()}"


def _post_prediction(client: TestClient) -> httpx.Response:
    resp: httpx.Response = client.post(
        "/api/v1/predict",
        json=dict(SAMPLE_INPUT),
        headers={"Idempotency-Key": _key()},
    )
    return resp


def _patch(
    client: TestClient,
    prediction_id: object,
    key: str,
    body: dict[str, object],
) -> httpx.Response:
    resp: httpx.Response = client.patch(
        f"/api/v1/predictions/{prediction_id}/outcome",
        json=body,
        headers={"Idempotency-Key": key},
    )
    return resp


def _patch_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "actual_churn": True,
        "expected_version": 0,
        "recorded_by": "analyst-a",
    }
    body.update(overrides)
    return body


def _stored(migrated_db_url: str, prediction_id: uuid.UUID) -> ChurnPrediction | None:
    engine = create_engine(migrated_db_url)
    try:
        with Session(engine) as session:
            return session.get(ChurnPrediction, prediction_id)
    finally:
        engine.dispose()


def test_patch_request_contract() -> None:
    """The DTO trims the author and rejects naive datetimes, blanks, extra
    fields and oversized authors (issue #10)."""
    parsed = OutcomePatchRequest(
        actual_churn=True,
        observed_at=datetime(2026, 2, 1, 12, 0, tzinfo=timezone(timedelta(hours=3))),
        expected_version=0,
        recorded_by="  analyst  ",
    )
    assert parsed.recorded_by == "analyst"
    assert parsed.observed_at is not None and parsed.observed_at.utcoffset()
    assert OutcomePatchRequest(
        actual_churn=False, expected_version=0, recorded_by="a"
    ).observed_at is None
    for invalid in (
        {"observed_at": "2026-02-01T12:00:00"},  # naive datetime
        {"recorded_by": "   "},  # blank author
        {"recorded_by": "x" * 129},  # oversized author
        {"origin": "demo_backfill"},  # origin is never client-chosen (D6)
        {"expected_version": -1},  # optimistic versions count from zero
    ):
        body = {**_patch_body(), **invalid}
        with pytest.raises(ValueError, match="Value error|extra|too_|greater_than"):
            OutcomePatchRequest.model_validate(body)


def test_retry_replays_byte_identical_without_reexecuting(
    predict_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _post_prediction(predict_client)
    assert created.status_code == 201
    prediction_id: str = created.json()["prediction_id"]
    key = _key()
    first = _patch(predict_client, prediction_id, key, _patch_body())
    assert first.status_code == 200

    def spy(*args: object, **kwargs: object) -> object:
        raise AssertionError("record_outcome must not run on a replay")

    monkeypatch.setattr("app.services.outcomes.record_outcome", spy)
    second = _patch(predict_client, prediction_id, key, _patch_body())
    assert second.status_code == 200
    assert second.json() == first.json()  # byte-identical replay (ADR-0014)
    body = first.json()
    assert body["prediction_id"] == prediction_id
    assert body["data_origin"] == "prospective"
    assert body["customer_id_ext"] is None  # sandbox POST: no identity
    assert body["outcome"]["actual_churn"] is True
    assert body["outcome"]["outcome_origin"] == "prospective"
    assert body["outcome"]["outcome_version"] == 1  # did not advance
    assert body["outcome"]["outcome_recorded_by"] == "analyst-a"


def test_same_key_different_payload_conflicts(predict_client: TestClient) -> None:
    prediction_id: str = _post_prediction(predict_client).json()["prediction_id"]
    key = _key()
    assert _patch(predict_client, prediction_id, key, _patch_body()).status_code == 200
    conflict = _patch(
        predict_client, prediction_id, key, _patch_body(recorded_by="analyst-b")
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"


def test_same_key_other_prediction_conflicts(predict_client: TestClient) -> None:
    """The operation is (key, prediction, payload): the same key against
    another prediction conflicts even with an equivalent body (triage B1)."""
    first_id: str = _post_prediction(predict_client).json()["prediction_id"]
    second_id: str = _post_prediction(predict_client).json()["prediction_id"]
    key = _key()
    assert _patch(predict_client, first_id, key, _patch_body()).status_code == 200
    conflict = _patch(predict_client, second_id, key, _patch_body())
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"


def test_stale_version_conflicts(predict_client: TestClient) -> None:
    prediction_id: str = _post_prediction(predict_client).json()["prediction_id"]
    conflict = _patch(
        predict_client, prediction_id, _key(), _patch_body(expected_version=1)
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "concurrency_conflict"


def test_immature_prospective_negative_conflicts_and_row_intact(
    predict_client: TestClient, migrated_db_url: str
) -> None:
    """A prospective negative before window maturity is a 409, never a
    bare 500; the stored prediction stays censored (ADR-0005)."""
    prediction_id: str = _post_prediction(predict_client).json()["prediction_id"]
    conflict = _patch(
        predict_client, prediction_id, _key(), _patch_body(actual_churn=False)
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "outcome_window_not_mature"
    stored = _stored(migrated_db_url, uuid.UUID(prediction_id))
    assert stored is not None
    assert stored.actual_churn is None
    assert stored.outcome_version == 0


def test_unknown_prediction_is_not_found(predict_client: TestClient) -> None:
    conflict = _patch(predict_client, uuid.uuid4(), _key(), _patch_body())
    assert conflict.status_code == 404
    assert conflict.json()["error"]["code"] == "prediction_not_found"


def test_mature_prospective_negative_is_accepted(
    predict_client: TestClient, migrated_db_url: str
) -> None:
    """A negative after the window matured is recorded (ADR-0005): the
    summary carries the identity, origin and the audit handle."""
    engine = create_engine(migrated_db_url)
    try:
        with Session(engine) as session:
            customer = register_customer(session, f"cust-{uuid.uuid4()}")
            score_at = datetime.now(UTC) - timedelta(days=91)
            prediction = record_prediction(
                session,
                customer_id=customer.id,
                batch_id=None,
                model_version="v1",
                churn_probability=0.6,
                decision_threshold=0.34,
                risk_band="high",
                factors={"method": "linear"},
                feature_snapshot={"tenure": 12},
                feature_schema_version="1",
                feature_schema_digest="a" * 64,
                factors_schema_digest="b" * 64,
                data_origin=DataOrigin.PROSPECTIVE,
                score_at=score_at,
                outcome_window_start=score_at,
                outcome_window_end=score_at + timedelta(days=90),
            )
            # Attributes expire on commit: capture what the assertions need
            # while the session is still open.
            prediction_id = prediction.id
            customer_id = customer.id
            customer_ext = customer.customer_id_ext
        response = _patch(
            predict_client,
            prediction_id,
            _key(),
            _patch_body(actual_churn=False, recorded_by="backfill"),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["customer_id_ext"] == customer_ext
        assert body["is_current"] is True
        outcome = body["outcome"]
        assert outcome["actual_churn"] is False
        assert outcome["outcome_origin"] == "prospective"
        assert outcome["outcome_recorded_by"] == "backfill"
        assert outcome["outcome_recorded_at"] is not None
        assert outcome["outcome_updated_at"] is None
        assert outcome["censored_at"] is None
        assert outcome["outcome_version"] == 1
    finally:
        # Committed rows leak into later tests against the shared session
        # database (e.g. pagination's customer count): remove them in FK
        # order.
        with Session(engine) as session:
            session.execute(
                sa.delete(OutcomeIdempotency).where(
                    OutcomeIdempotency.prediction_id == prediction_id
                )
            )
            session.execute(
                sa.delete(ChurnPrediction).where(
                    ChurnPrediction.id == prediction_id
                )
            )
            session.execute(sa.delete(Customer).where(Customer.id == customer_id))
            session.commit()
        engine.dispose()


def test_extra_field_is_rejected(predict_client: TestClient) -> None:
    """``extra="forbid"`` on the wire: the origin is never client-chosen
    (D6, ADR-0006) — an incompatible-origin attempt is a 422, not a
    silent discard."""
    prediction_id: str = _post_prediction(predict_client).json()["prediction_id"]
    resp = _patch(
        predict_client,
        prediction_id,
        _key(),
        {**_patch_body(), "origin": "demo_backfill"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_naive_observed_at_is_rejected(predict_client: TestClient) -> None:
    """The event moment must carry an explicit offset (issue #10): a naive
    datetime would be normalized from the process timezone and the
    canonical hash would drift between environments."""
    prediction_id: str = _post_prediction(predict_client).json()["prediction_id"]
    resp = _patch(
        predict_client,
        prediction_id,
        _key(),
        _patch_body(observed_at="2026-02-01T12:00:00"),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_idempotency_key_is_required(predict_client: TestClient) -> None:
    prediction_id: str = _post_prediction(predict_client).json()["prediction_id"]
    url = f"/api/v1/predictions/{prediction_id}/outcome"
    missing = predict_client.patch(url, json=_patch_body())
    blank = predict_client.patch(
        url, json=_patch_body(), headers={"Idempotency-Key": "   "}
    )
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "validation_error"
    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "validation_error"


def test_openapi_declares_the_patch_route(predict_client: TestClient) -> None:
    spec = predict_client.get("/api/openapi.json").json()
    patch = spec["paths"]["/api/v1/predictions/{prediction_id}/outcome"]["patch"]
    reference = patch["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ]
    assert reference.endswith("PredictionSummary")
    required_headers = [
        parameter
        for parameter in patch["parameters"]
        if parameter["in"] == "header" and parameter["required"]
    ]
    assert len(required_headers) == 1  # the Idempotency-Key (ADR-0014)
