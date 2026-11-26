"""Inference concurrency gate and transaction discipline (issue #9).

ADR-0012: inference/SHAP run in the threadpool behind an explicit
concurrency ceiling; excess requests fail fast with 429 instead of
queueing, health stays responsive, and no database session is held while
the CPU works (ADR-0014).
"""

import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.main import app
from app.models import ChurnPrediction, Customer
from app.schemas.explanation import TOP_K_DEFAULT, LocalExplanation
from app.services.explain import explain_local
from app.services.model_loader import LoadedModel
from tests.conftest import SAMPLE_INPUT


def _key() -> str:
    return f"test-{uuid.uuid4()}"


def _post(
    client: TestClient, key: str, customer: str | None = None
) -> httpx.Response:
    payload = dict(SAMPLE_INPUT)
    if customer is not None:
        payload["customer_id_ext"] = customer
    resp: httpx.Response = client.post(
        "/api/v1/predict", json=payload, headers={"Idempotency-Key": key}
    )
    return resp


def test_limit_is_enforced_and_excess_fails_fast(
    predict_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    limit = 2
    app.state.predict_semaphore = threading.BoundedSemaphore(limit)
    in_flight = 0
    max_seen = 0
    state_lock = threading.Lock()
    original = LoadedModel.predict_proba

    def slow(self: LoadedModel, features: Mapping[str, object]) -> float:
        nonlocal in_flight, max_seen
        with state_lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        try:
            time.sleep(0.4)
            return original(self, features)
        finally:
            with state_lock:
                in_flight -= 1

    monkeypatch.setattr(LoadedModel, "predict_proba", slow)
    with ThreadPoolExecutor(max_workers=4) as pool:
        clients = [TestClient(app) for _ in range(4)]
        responses = list(pool.map(_post, clients, [_key() for _ in range(4)]))
    assert max_seen <= limit
    assert sorted(resp.status_code for resp in responses) == [201, 201, 429, 429]


def test_health_stays_fast_while_inference_is_saturated(
    predict_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    app.state.predict_semaphore = threading.BoundedSemaphore(1)
    gate = threading.Event()
    started = threading.Event()
    original = LoadedModel.predict_proba

    def blocked(self: LoadedModel, features: Mapping[str, object]) -> float:
        started.set()
        assert gate.wait(timeout=10)
        return original(self, features)

    monkeypatch.setattr(LoadedModel, "predict_proba", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(_post, TestClient(app), _key())
        assert started.wait(timeout=10)
        health_start = time.perf_counter()
        health = TestClient(app).get("/api/v1/health/live")
        health_elapsed = time.perf_counter() - health_start
        rejected = _post(TestClient(app), _key())
        gate.set()
        result = pending.result()
    assert health.status_code == 200
    assert health_elapsed < 0.5
    assert rejected.status_code == 429
    assert result.status_code == 201


def test_no_connection_is_held_during_inference_or_shap(
    predict_client: TestClient, migrated_db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The predict service must open no session while scoring: checkouts
    on the engine are counted around the scorer and the explainer."""
    engine = create_engine(migrated_db_url)
    maker = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr("app.db.session.SessionLocal", maker)
    checkouts: list[float] = []

    def on_checkout(*args: object) -> None:
        checkouts.append(time.monotonic())

    event.listen(engine, "checkout", on_checkout)

    original_proba = LoadedModel.predict_proba

    def proba_spy(self: LoadedModel, features: Mapping[str, object]) -> float:
        before = len(checkouts)
        result = original_proba(self, features)
        assert len(checkouts) == before, "connection checked out during inference"
        return result

    monkeypatch.setattr(LoadedModel, "predict_proba", proba_spy)
    original_explain = explain_local

    def explain_spy(
        scorer: object,
        raw: Mapping[str, object],
        *,
        model_version: str,
        top_k: int = TOP_K_DEFAULT,
    ) -> LocalExplanation:
        before = len(checkouts)
        result = original_explain(scorer, raw, model_version=model_version, top_k=top_k)
        assert len(checkouts) == before, "connection checked out during SHAP"
        return result

    monkeypatch.setattr("app.services.predict.explain_local", explain_spy)
    try:
        resp = _post(predict_client, _key())
        assert resp.status_code == 201
    finally:
        engine.dispose()


def test_concurrent_distinct_customers_all_persist_consistently(
    predict_client: TestClient, migrated_db_url: str
) -> None:
    keys = [_key() for _ in range(4)]
    customers = [f"cust-{_key()}" for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(
                _post,
                [TestClient(predict_client.app)] * 4,
                keys,
                customers,
            )
        )
    assert all(resp.status_code == 201 for resp in responses)
    engine = create_engine(migrated_db_url)
    try:
        with Session(engine) as session:
            rows = session.scalars(
                select(ChurnPrediction).where(
                    ChurnPrediction.idempotency_key.in_(keys)
                )
            ).all()
            current_per_customer = session.scalar(
                select(func.count(ChurnPrediction.id))
                .where(ChurnPrediction.is_current.is_(True))
                .where(
                    ChurnPrediction.customer_id.in_(
                        select(Customer.id).where(
                            Customer.customer_id_ext.in_(customers)
                        )
                    )
                )
            )
    finally:
        engine.dispose()
    assert len(rows) == 4
    assert current_per_customer == 4  # one current prediction per customer
