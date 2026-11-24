"""Customers and origin-scoped service snapshots: constraints and services."""

import uuid
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Customer, CustomerService, DataOrigin
from app.services.customers import register_customer, upsert_service_snapshot


def _customer(session: Session, ext_id: str = "C-1") -> Customer:
    customer = Customer(customer_id_ext=ext_id)
    session.add(customer)
    session.flush()
    return customer


def test_blank_customer_id_ext_rejected(db_session: Session) -> None:
    db_session.add(Customer(customer_id_ext="   "))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_untrimmed_customer_id_ext_rejected(db_session: Session) -> None:
    db_session.add(Customer(customer_id_ext=" C-1"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_duplicate_customer_id_ext_rejected(db_session: Session) -> None:
    _customer(db_session, "C-DUP")
    db_session.add(Customer(customer_id_ext="C-DUP"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_register_customer_is_get_or_create(db_session: Session) -> None:
    first = register_customer(db_session, "  C-42  ")
    second = register_customer(db_session, "C-42")
    assert first.id == second.id
    assert first.customer_id_ext == "C-42"


def test_service_cardinality_is_explicit(db_session: Session) -> None:
    customer = _customer(db_session)
    for service_type in ("phone", "internet"):
        db_session.add(
            CustomerService(
                customer_id=customer.id,
                data_origin=DataOrigin.DEMO_BACKFILL,
                service_type=service_type,
            )
        )
    # Same service type may coexist under a different origin.
    db_session.add(
        CustomerService(
            customer_id=customer.id,
            data_origin=DataOrigin.PROSPECTIVE,
            service_type="phone",
        )
    )
    db_session.flush()
    count = db_session.scalar(
        sa.select(sa.func.count())
        .select_from(CustomerService)
        .where(CustomerService.customer_id == customer.id)
    )
    assert count == 3


def test_second_current_snapshot_same_scope_rejected(db_session: Session) -> None:
    customer = _customer(db_session)
    for _ in range(2):
        db_session.add(
            CustomerService(
                customer_id=customer.id,
                data_origin=DataOrigin.DEMO_BACKFILL,
                service_type="phone",
            )
        )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_validity_window_order_enforced(db_session: Session) -> None:
    customer = _customer(db_session)
    now = datetime.now(UTC)
    db_session.add(
        CustomerService(
            customer_id=customer.id,
            data_origin=DataOrigin.DEMO_BACKFILL,
            service_type="phone",
            valid_from=now,
            valid_to=now,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_attributes_must_be_object(db_session: Session) -> None:
    customer = _customer(db_session)
    # A JSON array is not a valid snapshot payload.
    with pytest.raises(IntegrityError):
        db_session.execute(
            sa.text(
                "INSERT INTO customer_services"
                " (customer_id, data_origin, service_type, attributes)"
                " VALUES (:cid, 'demo_backfill', 'phone', '[1, 2]'::jsonb)"
            ),
            {"cid": customer.id},
        )


def test_upsert_closes_previous_snapshot_and_keeps_history(
    db_session: Session,
) -> None:
    customer = register_customer(db_session, "C-77")
    first = upsert_service_snapshot(
        db_session,
        customer_id=customer.id,
        data_origin=DataOrigin.DEMO_BACKFILL,
        service_type="internet",
        attributes={"plan": "fiber"},
    )
    second = upsert_service_snapshot(
        db_session,
        customer_id=customer.id,
        data_origin=DataOrigin.DEMO_BACKFILL,
        service_type="internet",
        attributes={"plan": "fiber-2"},
    )
    db_session.expire_all()
    assert first.valid_to is not None
    assert second.valid_to is None
    rows = db_session.scalars(
        sa.select(CustomerService).where(
            CustomerService.customer_id == customer.id,
            CustomerService.service_type == "internet",
        )
    ).all()
    assert len(rows) == 2  # history preserved, nothing overwritten


def test_upsert_does_not_overwrite_other_origin(db_session: Session) -> None:
    customer = register_customer(db_session, "C-78")
    prospective = upsert_service_snapshot(
        db_session,
        customer_id=customer.id,
        data_origin=DataOrigin.PROSPECTIVE,
        service_type="phone",
        attributes={"plan": "real"},
    )
    upsert_service_snapshot(
        db_session,
        customer_id=customer.id,
        data_origin=DataOrigin.DEMO_BACKFILL,
        service_type="phone",
        attributes={"plan": "demo"},
    )
    db_session.expire_all()
    assert prospective.valid_to is None  # demo upsert left prospective current
    current_origins = db_session.scalars(
        sa.select(CustomerService.data_origin).where(
            CustomerService.customer_id == customer.id,
            CustomerService.valid_to.is_(None),
        )
    ).all()
    assert set(current_origins) == {
        DataOrigin.PROSPECTIVE,
        DataOrigin.DEMO_BACKFILL,
    }


def test_unknown_customer_service_fk_rejected(db_session: Session) -> None:
    db_session.add(
        CustomerService(
            customer_id=uuid.uuid4(),
            data_origin=DataOrigin.DEMO_BACKFILL,
            service_type="phone",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
