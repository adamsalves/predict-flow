"""outcome idempotency registry

Revision ID: a00785fe6576
Revises: d78974925de2
Create Date: 2026-11-27 10:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a00785fe6576"
down_revision: str | None = "d78974925de2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "outcome_idempotency",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("prediction_id", sa.Uuid(), nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.Column("resulted_version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "payload_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_outcome_idempotency_payload_sha256_hex"),
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["churn_predictions.id"],
            name=op.f("fk_outcome_idempotency_prediction_id_churn_predictions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_outcome_idempotency")),
        # Explicit name: the service resolves the concurrent first-write
        # race by reading the driver-reported constraint name (ADR-0014).
        sa.UniqueConstraint("idempotency_key", name="uq_outcome_idempotency_key"),
    )


def downgrade() -> None:
    op.drop_table("outcome_idempotency")
