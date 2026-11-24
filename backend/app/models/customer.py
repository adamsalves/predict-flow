"""Customer identity (issue #7).

``customers`` holds only the global identity of a customer: the external id
is unique across the whole database and mutable operational state never
lives here. Operational snapshots are origin-scoped rows in
``customer_services`` (ADR-0015), so an upsert from one origin can never
silently overwrite the state of another origin for the same identity.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Customer(Base):
    __tablename__ = "customers"
    __table_args__ = (
        sa.UniqueConstraint("customer_id_ext", name="uq_customers_customer_id_ext"),
        # Normalized means trimmed and non-empty; the database rejects raw
        # values that would make lookups ambiguous.
        sa.CheckConstraint(
            "customer_id_ext <> '' AND customer_id_ext = btrim(customer_id_ext)",
            name="customer_id_ext_normalized",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid, primary_key=True, server_default=sa.func.gen_random_uuid()
    )
    customer_id_ext: Mapped[str] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()
    )
