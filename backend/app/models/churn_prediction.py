"""Append-only churn prediction history with explicit currency (issue #7).

History is immutable: a new prediction never overwrites older rows
(ADR-0015). The current prediction per (customer, origin) is marked by
``is_current`` and enforced by a partial unique index — the "equivalent
constraint" allowed by ADR-0015 in place of a separate projection table.
Anonymous sandbox predictions have ``customer_id NULL``, are never current,
and therefore never enter the customer worklist.

Outcome fields follow ADR-0005: ``actual_churn`` is born NULL (censored,
never a confirmed negative), every update records origin, author and
observation/record moments, and a prospective ``actual_churn = false`` is
only accepted after the outcome window has matured. Fields the Telco
dataset cannot support (``feature_as_of``, outcome observation moments)
stay NULL and that limitation is explicit — historical labels are never
promoted to mature prospective outcomes.

FK behavior: ``customer_id`` is ``ON DELETE RESTRICT`` — prediction history
blocks identity deletion. ``batch_id`` stays a plain UUID until the batch
table lands with issue #18.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, DataOrigin, JsonValue, data_origin_enum

# Payloads are auditable snapshots, not blobs: objects only, bounded size.
_MAX_PAYLOAD_BYTES = 16384
_DIGEST_HEX_PATTERN = "^[0-9a-f]{64}$"


class ChurnPrediction(Base):
    __tablename__ = "churn_predictions"
    __table_args__ = (
        sa.CheckConstraint(
            "model_version <> '' AND model_version = btrim(model_version)",
            name="model_version_normalized",
        ),
        sa.CheckConstraint(
            "churn_probability BETWEEN 0 AND 1", name="churn_probability_range"
        ),
        sa.CheckConstraint(
            "decision_threshold IS NULL OR decision_threshold BETWEEN 0 AND 1",
            name="decision_threshold_range",
        ),
        # Band and threshold come from the same scoring-time policy: both
        # present or both absent.
        sa.CheckConstraint(
            "(risk_band IS NULL) = (decision_threshold IS NULL)",
            name="risk_band_threshold_coupled",
        ),
        sa.CheckConstraint("risk_band IN ('high', 'low')", name="risk_band_values"),
        sa.CheckConstraint("jsonb_typeof(factors) = 'object'", name="factors_object"),
        sa.CheckConstraint(
            f"pg_column_size(factors) <= {_MAX_PAYLOAD_BYTES}", name="factors_size"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(feature_snapshot) = 'object'",
            name="feature_snapshot_object",
        ),
        sa.CheckConstraint(
            f"pg_column_size(feature_snapshot) <= {_MAX_PAYLOAD_BYTES}",
            name="feature_snapshot_size",
        ),
        sa.CheckConstraint(
            f"feature_schema_digest ~ '{_DIGEST_HEX_PATTERN}'",
            name="feature_schema_digest_hex",
        ),
        sa.CheckConstraint(
            f"factors_schema_digest ~ '{_DIGEST_HEX_PATTERN}'",
            name="factors_schema_digest_hex",
        ),
        # Anti-leakage (ADR-0009): features must not come from the future.
        sa.CheckConstraint(
            "feature_as_of IS NULL OR feature_as_of <= score_at",
            name="feature_as_of_not_after_score",
        ),
        sa.CheckConstraint(
            "(outcome_window_start IS NULL) = (outcome_window_end IS NULL)",
            name="outcome_window_pair",
        ),
        sa.CheckConstraint(
            "outcome_window_start IS NULL OR outcome_window_end > outcome_window_start",
            name="outcome_window_order",
        ),
        # No orphan outcome metadata: everything is NULL until an outcome
        # exists; origin, author and record moment are mandatory with one.
        sa.CheckConstraint(
            "actual_churn IS NOT NULL OR (outcome_origin IS NULL"
            " AND outcome_recorded_by IS NULL AND outcome_observed_at IS NULL"
            " AND outcome_recorded_at IS NULL AND outcome_updated_at IS NULL)",
            name="outcome_metadata_orphan",
        ),
        sa.CheckConstraint(
            "actual_churn IS NULL OR (outcome_origin IS NOT NULL"
            " AND outcome_recorded_by IS NOT NULL"
            " AND outcome_recorded_at IS NOT NULL)",
            name="outcome_bundle_complete",
        ),
        sa.CheckConstraint(
            "outcome_updated_at IS NULL OR outcome_updated_at >= outcome_recorded_at",
            name="outcome_update_order",
        ),
        # Censoring marks the absence of an outcome; both cannot coexist.
        sa.CheckConstraint(
            "censored_at IS NULL OR actual_churn IS NULL",
            name="censored_without_outcome",
        ),
        # ADR-0005: a prospective "did not churn" is only accepted after the
        # outcome window has fully matured at record time.
        sa.CheckConstraint(
            "NOT (data_origin = 'prospective' AND actual_churn IS FALSE)"
            " OR (outcome_window_end IS NOT NULL AND outcome_recorded_at IS NOT NULL"
            " AND outcome_recorded_at >= outcome_window_end)",
            name="prospective_negative_maturity",
        ),
        # Anonymous sandbox predictions never join the current projection.
        sa.CheckConstraint(
            "customer_id IS NOT NULL OR NOT is_current", name="sandbox_not_current"
        ),
        # Single current prediction per (customer, origin); sandbox rows
        # (NULL customer) are outside the projection by construction.
        sa.Index(
            "uq_churn_predictions_current",
            "customer_id",
            "data_origin",
            unique=True,
            postgresql_where=sa.text("is_current"),
        ),
        # History per customer (optionally filtered by origin), time-ordered.
        sa.Index(
            "ix_churn_predictions_customer_history",
            "customer_id",
            "data_origin",
            "score_at",
        ),
        sa.Index(
            "ix_churn_predictions_batch",
            "batch_id",
            postgresql_where=sa.text("batch_id IS NOT NULL"),
        ),
        # KPI queries always filter by origin (ADR-0006).
        sa.Index("ix_churn_predictions_origin_score", "data_origin", "score_at"),
        # Worklist: current predictions per origin ranked by probability,
        # with the stable id tiebreak (ADR-0015).
        sa.Index(
            "ix_churn_predictions_worklist",
            "data_origin",
            "churn_probability",
            "id",
            postgresql_where=sa.text("is_current"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid, primary_key=True, server_default=sa.func.gen_random_uuid()
    )
    customer_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("customers.id", ondelete="RESTRICT")
    )
    batch_id: Mapped[uuid.UUID | None] = mapped_column(sa.Uuid)
    model_version: Mapped[str] = mapped_column(sa.Text)
    churn_probability: Mapped[float] = mapped_column(sa.Float)
    decision_threshold: Mapped[float | None] = mapped_column(sa.Float)
    risk_band: Mapped[str | None] = mapped_column(sa.Text)
    factors: Mapped[dict[str, JsonValue]] = mapped_column(JSONB)
    feature_snapshot: Mapped[dict[str, JsonValue]] = mapped_column(JSONB)
    feature_schema_version: Mapped[str] = mapped_column(sa.Text)
    feature_schema_digest: Mapped[str] = mapped_column(sa.Text)
    factors_schema_digest: Mapped[str] = mapped_column(sa.Text)
    data_origin: Mapped[DataOrigin] = mapped_column(data_origin_enum("data_origin"))
    score_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    feature_as_of: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    outcome_window_start: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True)
    )
    outcome_window_end: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True)
    )
    actual_churn: Mapped[bool | None] = mapped_column(sa.Boolean)
    outcome_origin: Mapped[DataOrigin | None] = mapped_column(
        data_origin_enum("outcome_origin")
    )
    outcome_recorded_by: Mapped[str | None] = mapped_column(sa.Text)
    outcome_observed_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True)
    )
    outcome_recorded_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True)
    )
    outcome_updated_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True)
    )
    censored_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    outcome_version: Mapped[int] = mapped_column(
        sa.Integer, server_default=sa.text("0")
    )
    is_current: Mapped[bool] = mapped_column(
        sa.Boolean, server_default=sa.false()
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
