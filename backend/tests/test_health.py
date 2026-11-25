import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import database_available


def test_live(client: TestClient) -> None:
    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_sem_db_retorna_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.v1.health.database_available", lambda: False)
    resp = client.get("/api/v1/health/ready")
    assert resp.status_code == 503


def test_ready_com_db_retorna_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.v1.health.database_available", lambda: True)
    resp = client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_ready_com_db_real(client: TestClient) -> None:
    if not database_available():
        pytest.skip("Postgres indisponível neste ambiente")
    resp = client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


# TEST-NET-1 (RFC 5737): guaranteed unroutable — the connect attempt hangs
# until the explicit timeout fires (or fails fast with no route).
BLACKHOLE_URL = "postgresql+psycopg://predict:predict@192.0.2.1:5432/predict_flow"


@pytest.fixture
def blackholed_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", BLACKHOLE_URL)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def test_database_available_blackholed_db_is_bounded(
    blackholed_db: None,
) -> None:
    start = time.perf_counter()
    available = database_available()
    elapsed = time.perf_counter() - start
    assert available is False
    # Default budget is 2s connect + 2s statement; unbounded TCP connect
    # would take minutes.
    assert elapsed < 8.0


def test_ready_blackholed_db_returns_503_within_timeout(
    client: TestClient, blackholed_db: None
) -> None:
    start = time.perf_counter()
    resp = client.get("/api/v1/health/ready")
    elapsed = time.perf_counter() - start
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "service_unavailable"
    assert elapsed < 8.0
