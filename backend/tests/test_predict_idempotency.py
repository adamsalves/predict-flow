"""Idempotency and persistence failure behavior of POST /predict (issue #9).

ADR-0014: the key binds one canonical payload to one recorded result —
an equivalent retry replays without duplicating, a different payload
conflicts, and a persistence failure never returns an audited success.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.models import ChurnPrediction
from app.services import predictions as predictions_service
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


def _count_by_key(url: str, key: str) -> int:
    engine = create_engine(url)
    try:
        with Session(engine) as session:
            return int(
                session.scalar(
                    select(func.count(ChurnPrediction.id)).where(
                        ChurnPrediction.idempotency_key == key
                    )
                )
                or 0
            )
    finally:
        engine.dispose()


def test_equivalent_retry_replays_without_duplicating(
    predict_client: TestClient, migrated_db_url: str
) -> None:
    key = _key()
    first = _predict(predict_client, dict(SAMPLE_INPUT), key)
    assert first.status_code == 201
    replay = _predict(predict_client, dict(SAMPLE_INPUT), key)
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert _count_by_key(migrated_db_url, key) == 1


def test_semantically_shuffled_payload_still_replays(
    predict_client: TestClient, migrated_db_url: str
) -> None:
    """Canonical hashing is order-insensitive: the same row scored from a
    JSON object with keys in a different order is the same operation."""
    key = _key()
    first = _predict(predict_client, dict(SAMPLE_INPUT), key)
    assert first.status_code == 201
    shuffled = dict(reversed(list(SAMPLE_INPUT.items())))
    replay = _predict(predict_client, shuffled, key)
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert _count_by_key(migrated_db_url, key) == 1


def test_negative_zero_replays_as_positive_zero(
    predict_client: TestClient, migrated_db_url: str
) -> None:
    """-0.0 and 0.0 are the same charge: the schema normalizes the sign, so
    the canonical hash matches and the retry replays instead of 409."""
    key = _key()
    first = _predict(
        predict_client, {**SAMPLE_INPUT, "MonthlyCharges": -0.0}, key
    )
    assert first.status_code == 201
    replay = _predict(
        predict_client, {**SAMPLE_INPUT, "MonthlyCharges": 0.0}, key
    )
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert _count_by_key(migrated_db_url, key) == 1


def test_same_key_with_different_payload_conflicts(
    predict_client: TestClient, migrated_db_url: str
) -> None:
    key = _key()
    assert _predict(predict_client, dict(SAMPLE_INPUT), key).status_code == 201
    resp = _predict(
        predict_client, {**SAMPLE_INPUT, "tenure": 24}, key
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "idempotency_conflict"
    assert _count_by_key(migrated_db_url, key) == 1


def test_different_key_records_new_prediction_preserving_history(
    predict_client: TestClient, migrated_db_url: str
) -> None:
    customer = f"cust-{_key()}"
    first_key, second_key = _key(), _key()
    first = _predict(
        predict_client, {**SAMPLE_INPUT, "customer_id_ext": customer}, first_key
    )
    second = _predict(
        predict_client, {**SAMPLE_INPUT, "customer_id_ext": customer}, second_key
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["prediction_id"] != second.json()["prediction_id"]
    engine = create_engine(migrated_db_url)
    try:
        with Session(engine) as session:
            rows = session.scalars(
                select(ChurnPrediction)
                .where(ChurnPrediction.idempotency_key.in_([first_key, second_key]))
                .order_by(ChurnPrediction.score_at)
            ).all()
    finally:
        engine.dispose()
    assert len(rows) == 2
    assert rows[0].is_current is False  # history preserved (ADR-0015)
    assert rows[1].is_current is True


def test_concurrent_same_key_posts_leave_exactly_one_row(
    predict_client: TestClient, migrated_db_url: str
) -> None:
    key = _key()
    # Each thread gets its own TestClient: same app, same app.state, but
    # no shared httpx client across threads.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _predict, TestClient(predict_client.app), dict(SAMPLE_INPUT), key
            )
            for _ in range(2)
        ]
        bodies = [future.result() for future in futures]
    assert all(resp.status_code in (200, 201) for resp in bodies)
    assert bodies[0].json() == bodies[1].json()
    assert _count_by_key(migrated_db_url, key) == 1


def test_persistence_failure_is_not_an_audited_success_and_retry_is_safe(
    predict_client: TestClient,
    migrated_db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _key()

    def boom(*args: object, **kwargs: object) -> ChurnPrediction:
        raise RuntimeError("database is gone")

    original = predictions_service.record_prediction
    monkeypatch.setattr("app.services.predict.record_prediction", boom)
    failed = _predict(predict_client, dict(SAMPLE_INPUT), key)
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "internal_error"
    assert _count_by_key(migrated_db_url, key) == 0

    # Restore only this attribute (monkeypatch.undo would also drop the
    # fixture's SessionLocal repoint and env overrides).
    monkeypatch.setattr("app.services.predict.record_prediction", original)
    retry = _predict(predict_client, dict(SAMPLE_INPUT), key)
    assert retry.status_code == 201
    assert _count_by_key(migrated_db_url, key) == 1
    final = _predict(predict_client, dict(SAMPLE_INPUT), key)
    assert final.status_code == 200
    assert final.json() == retry.json()
