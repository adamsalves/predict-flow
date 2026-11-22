import pytest
from fastapi.testclient import TestClient

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
