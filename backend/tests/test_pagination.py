"""Pagination convention: bounded limit, stable order, unique tie-break (#40)."""

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.pagination import page_params
from app.db.pagination import paginate
from app.models import Customer
from app.schemas.pagination import MAX_PAGE_SIZE, PageParams
from tests.conftest import build_conventions_app


def _params_app() -> TestClient:
    app = build_conventions_app()

    @app.get("/rows")
    def rows(params: Annotated[PageParams, Depends(page_params)]) -> dict[str, int]:
        return {"limit": params.limit, "offset": params.offset}

    return TestClient(app)


def test_page_params_defaults_and_bounds() -> None:
    client = _params_app()
    assert client.get("/rows").json() == {"limit": 20, "offset": 0}
    assert client.get("/rows", params={"limit": 100, "offset": 5}).json() == {
        "limit": 100,
        "offset": 5,
    }
    for params in ({"limit": 0}, {"limit": MAX_PAGE_SIZE + 1}, {"offset": -1}):
        resp = client.get("/rows", params=params)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"


def _seed_customers(session: Session) -> list[Customer]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    # Ties are forced: three customers share T1, two share T2, two share T3.
    stamps = [base, base, base, base.replace(day=2), base.replace(day=2),
              base.replace(day=3), base.replace(day=3)]
    customers = [
        Customer(customer_id_ext=f"cust-{index}", created_at=stamp)
        for index, stamp in enumerate(stamps)
    ]
    session.add_all(customers)
    session.flush()
    return customers


def test_paginate_is_stable_under_tied_sort_keys(db_session: Session) -> None:
    customers = _seed_customers(db_session)
    expected_ids = {customer.id for customer in customers}

    def collect() -> list[uuid.UUID]:
        seen: list[uuid.UUID] = []
        for offset in (0, 3, 6):
            page, total = paginate(
                db_session,
                select(Customer),
                order_by=(Customer.created_at,),
                tiebreak=Customer.id,
                params=PageParams(limit=3, offset=offset),
            )
            assert total == 7
            seen.extend(customer.id for customer in page)
        return seen

    first, second = collect(), collect()
    # No repeats, no skips, and identical order across repeated scans.
    assert len(first) == 7
    assert set(first) == expected_ids
    assert first == second


def test_paginate_descending_order_is_also_stable(db_session: Session) -> None:
    _seed_customers(db_session)
    page, total = paginate(
        db_session,
        select(Customer),
        order_by=(Customer.created_at.desc(),),
        tiebreak=Customer.id,
        params=PageParams(limit=100, offset=0),
    )
    assert total == 7
    assert len(page) == 7
    stamps = [customer.created_at for customer in page]
    assert stamps == sorted(stamps, reverse=True)
    tied = [c.id for c in page if c.created_at == stamps[0]]
    assert tied == sorted(tied)


def test_paginate_beyond_last_page(db_session: Session) -> None:
    _seed_customers(db_session)
    page, total = paginate(
        db_session,
        select(Customer),
        order_by=(Customer.created_at,),
        tiebreak=Customer.id,
        params=PageParams(limit=3, offset=100),
    )
    assert page == []
    assert total == 7
