"""Session-per-request and transversal rollback at the HTTP boundary (#40).

The production ``get_session`` dependency runs unchanged — only
``SessionLocal`` is repointed to a throwaway database — so these tests
exercise the real wiring: one session per request, always closed, and
uncommitted work rolled back on success and on failure alike (ADR-0012).
"""

import uuid
from collections.abc import Callable, Iterator

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import get_session
from app.models import Customer
from app.services.errors import DomainError
from tests.conftest import build_conventions_app


@pytest.fixture
def session_app(
    migrated_db_url: str, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, list[Session], Callable[[], int]]]:
    engine = create_engine(migrated_db_url)
    maker = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr("app.db.session.SessionLocal", maker)
    seen: list[Session] = []
    app = build_conventions_app()

    @app.post("/customers")
    def create_customer(session: Session = Depends(get_session)) -> dict[str, str]:
        seen.append(session)
        session.add(Customer(customer_id_ext=f"c-{uuid.uuid4().hex[:8]}"))
        return {"status": "ok"}  # no commit: closing must roll back

    @app.post("/customers-fail")
    def create_customer_fail(session: Session = Depends(get_session)) -> None:
        seen.append(session)
        session.add(Customer(customer_id_ext=f"c-{uuid.uuid4().hex[:8]}"))
        raise DomainError("boom")

    def count_customers() -> int:
        with maker() as session:
            return int(session.scalar(select(func.count(Customer.id))) or 0)

    try:
        yield TestClient(app), seen, count_customers
    finally:
        engine.dispose()


def test_each_request_gets_a_fresh_closed_session(
    session_app: tuple[TestClient, list[Session], Callable[[], int]],
) -> None:
    client, seen, _ = session_app
    assert client.post("/customers").status_code == 200
    assert client.post("/customers").status_code == 200
    assert len(seen) == 2
    assert seen[0] is not seen[1]
    assert not seen[0].in_transaction()
    assert not seen[1].in_transaction()


def test_uncommitted_work_is_rolled_back(
    session_app: tuple[TestClient, list[Session], Callable[[], int]],
) -> None:
    client, _, count_customers = session_app
    before = count_customers()
    assert client.post("/customers").status_code == 200
    assert count_customers() == before


def test_exception_mid_request_rolls_back_and_uses_envelope(
    session_app: tuple[TestClient, list[Session], Callable[[], int]],
) -> None:
    client, _, count_customers = session_app
    before = count_customers()
    resp = client.post("/customers-fail")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "domain_error"
    assert count_customers() == before
