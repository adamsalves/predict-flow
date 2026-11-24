"""Origin-scoped customer service snapshots (issue #7).

A customer has an explicit cardinality of service rows, each with its own
identity. Every row belongs to exactly one ``data_origin`` and carries a
validity window: closing a snapshot sets ``valid_to`` instead of updating
in place, so current features are never mixed with the state that fed an
older prediction. A partial unique index guarantees at most one current
snapshot per (customer, origin, service type) — an upsert from one origin
cannot overwrite another origin's state (ADR-0015).

FK behavior: ``customer_id`` is ``ON DELETE RESTRICT`` — deleting an
identity while snapshots exist is blocked, preserving history.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, DataOrigin, JsonValue, data_origin_enum


class CustomerService(Base):
    __tablename__ = "customer_services"
    __table_args__ = (
        sa.CheckConstraint(
            "service_type <> '' AND service_type = btrim(service_type)",
            name="service_type_normalized",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(attributes) = 'object'", name="attributes_object"
        ),
        sa.CheckConstraint(
            "valid_to IS NULL OR valid_to > valid_from", name="validity_window_order"
        ),
        # One current snapshot per (customer, origin, service type).
        sa.Index(
            "uq_customer_services_current",
            "customer_id",
            "data_origin",
            "service_type",
            unique=True,
            postgresql_where=sa.text("valid_to IS NULL"),
        ),
        # Snapshot history per customer and origin.
        sa.Index(
            "ix_customer_services_customer_history",
            "customer_id",
            "data_origin",
            "valid_from",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid, primary_key=True, server_default=sa.func.gen_random_uuid()
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("customers.id", ondelete="RESTRICT")
    )
    data_origin: Mapped[DataOrigin] = mapped_column(data_origin_enum("data_origin"))
    service_type: Mapped[str] = mapped_column(sa.Text)
    attributes: Mapped[dict[str, JsonValue]] = mapped_column(
        JSONB, server_default=sa.text("'{}'::jsonb")
    )
    valid_from: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
    valid_to: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
