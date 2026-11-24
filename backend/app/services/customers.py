"""Customer identity and origin-scoped service snapshots (issue #7).

Each function delimits its own transaction (ADR-0012). The partial unique
index on current snapshots is the final race protection (ADR-0014): a
concurrent upsert loses at commit with an integrity error instead of
silently overwriting state.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Customer, CustomerService, DataOrigin, JsonValue


def register_customer(session: Session, customer_id_ext: str) -> Customer:
    """Get-or-create the global identity for an external customer id.

    The database enforces normalization (trimmed, non-empty) and global
    uniqueness; a concurrent insert of the same id is absorbed by the
    unique constraint and resolved by re-reading the existing row.
    """
    normalized = customer_id_ext.strip()
    existing = session.scalar(
        select(Customer).where(Customer.customer_id_ext == normalized)
    )
    if existing is not None:
        return existing
    customer = Customer(customer_id_ext=normalized)
    session.add(customer)
    try:
        with session.begin_nested():
            session.flush()
    except IntegrityError:
        # The savepoint is already rolled back by begin_nested; re-read the
        # row committed by the concurrent transaction.
        found = session.scalar(
            select(Customer).where(Customer.customer_id_ext == normalized)
        )
        if found is None:
            raise
        return found
    session.commit()
    return customer


def upsert_service_snapshot(
    session: Session,
    *,
    customer_id: uuid.UUID,
    data_origin: DataOrigin,
    service_type: str,
    attributes: dict[str, JsonValue],
) -> CustomerService:
    """Replace the current snapshot for (customer, origin, service type).

    The previous snapshot is closed (``valid_to`` set) instead of updated,
    preserving the state that fed older predictions; the new snapshot only
    supersedes the same origin — other origins keep their own currency
    (ADR-0015).
    """
    now = datetime.now(UTC)
    current = session.scalar(
        select(CustomerService).where(
            CustomerService.customer_id == customer_id,
            CustomerService.data_origin == data_origin,
            CustomerService.service_type == service_type,
            CustomerService.valid_to.is_(None),
        )
    )
    if current is not None:
        current.valid_to = now
    snapshot = CustomerService(
        customer_id=customer_id,
        data_origin=data_origin,
        service_type=service_type,
        attributes=attributes,
        valid_from=now,
    )
    session.add(snapshot)
    session.commit()
    return snapshot
