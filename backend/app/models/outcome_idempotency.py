"""Idempotency registry of outcome operations (issue #10, ADR-0014).

One row per idempotent outcome call: the client-supplied key binds one
canonical payload hash to the prediction and the outcome version the
operation produced. An equivalent retry replays the stored result without
re-executing the update; the same key with a different payload conflicts.
The unique constraint on the key — with an explicit name, read from the
driver's diagnostic when violated — is the final arbiter of concurrent
first writes (ADR-0014): the database, not application locking, decides
the winner.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

# Payload digest is a lowercase SHA-256 hex string (same style as the
# prediction payload hash of issue #9).
_DIGEST_HEX_PATTERN = "^[0-9a-f]{64}$"


class OutcomeIdempotency(Base):
    __tablename__ = "outcome_idempotency"

    __table_args__ = (
        sa.CheckConstraint(
            f"payload_sha256 ~ '{_DIGEST_HEX_PATTERN}'",
            name="payload_sha256_hex",
        ),
        sa.UniqueConstraint("idempotency_key", name="uq_outcome_idempotency_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid, primary_key=True, server_default=sa.func.gen_random_uuid()
    )
    idempotency_key: Mapped[str] = mapped_column(sa.Text)
    # The operation the key produced; RESTRICT keeps the audited registry
    # alive as long as the prediction history exists (ADR-0015).
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("churn_predictions.id", ondelete="RESTRICT")
    )
    payload_sha256: Mapped[str] = mapped_column(sa.Text)
    resulted_version: Mapped[int] = mapped_column(sa.Integer)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
