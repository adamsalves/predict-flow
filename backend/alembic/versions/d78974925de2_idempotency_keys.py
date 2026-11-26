"""idempotency keys for predictions

Revision ID: d78974925de2
Revises: 72b406a0c873
Create Date: 2026-11-26 09:30:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d78974925de2"
down_revision: str | None = "72b406a0c873"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "churn_predictions",
        sa.Column("idempotency_key", sa.Text(), nullable=True),
    )
    op.add_column(
        "churn_predictions",
        sa.Column("request_payload_sha256", sa.Text(), nullable=True),
    )
    op.create_check_constraint(
        "idempotency_key_normalized",
        "churn_predictions",
        "idempotency_key IS NULL"
        " OR (idempotency_key <> ''"
        " AND idempotency_key = btrim(idempotency_key))",
    )
    op.create_check_constraint(
        "request_payload_sha256_hex",
        "churn_predictions",
        "request_payload_sha256 IS NULL"
        " OR request_payload_sha256 ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "idempotency_key_payload_paired",
        "churn_predictions",
        "(idempotency_key IS NULL) = (request_payload_sha256 IS NULL)",
    )
    op.create_index(
        "uq_churn_predictions_idempotency_key",
        "churn_predictions",
        ["idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_churn_predictions_idempotency_key", table_name="churn_predictions"
    )
    op.drop_constraint(
        "idempotency_key_payload_paired",
        "churn_predictions",
        type_="check",
    )
    op.drop_constraint(
        "request_payload_sha256_hex",
        "churn_predictions",
        type_="check",
    )
    op.drop_constraint(
        "idempotency_key_normalized",
        "churn_predictions",
        type_="check",
    )
    op.drop_column("churn_predictions", "request_payload_sha256")
    op.drop_column("churn_predictions", "idempotency_key")
