"""Prediction recording, currency activation and outcome updates (issue #7).

History is append-only (ADR-0015): recording a prediction inserts a new
row and, in the same transaction, moves the current marker for
(customer, origin) to it — a failure anywhere leaves the previous current
prediction untouched. Outcome updates use optimistic concurrency
(ADR-0014): an update based on a stale ``outcome_version`` conflicts
instead of silently losing a correction.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import case, select, update
from sqlalchemy.orm import Session

from app.models import ChurnPrediction, DataOrigin, JsonValue
from app.services.errors import ConcurrencyConflictError


def find_by_idempotency_key(session: Session, key: str) -> ChurnPrediction | None:
    """Fetch the recorded result of a previous call with this key (issue #9)."""
    return session.scalar(
        select(ChurnPrediction).where(ChurnPrediction.idempotency_key == key)
    )


def record_prediction(
    session: Session,
    *,
    customer_id: uuid.UUID | None,
    batch_id: uuid.UUID | None,
    model_version: str,
    churn_probability: float,
    decision_threshold: float | None,
    risk_band: str | None,
    factors: dict[str, JsonValue],
    feature_snapshot: dict[str, JsonValue],
    feature_schema_version: str,
    feature_schema_digest: str,
    factors_schema_digest: str,
    data_origin: DataOrigin,
    score_at: datetime,
    feature_as_of: datetime | None = None,
    outcome_window_start: datetime | None = None,
    outcome_window_end: datetime | None = None,
    idempotency_key: str | None = None,
    request_payload_sha256: str | None = None,
    activate: bool = True,
) -> ChurnPrediction:
    """Insert a prediction and atomically move the current marker.

    Sandbox calls (``customer_id=None``) never join the current projection
    and never enter the customer worklist. A concurrent activation for the
    same (customer, origin) loses at commit on the partial unique index —
    the database, not application locking, is the final arbiter (ADR-0014).
    ``idempotency_key``/``request_payload_sha256`` bind the row to one API
    call (issue #9); a concurrent insert with the same key raises
    ``IntegrityError`` at commit, which the caller resolves into a replay
    or a conflict.
    """
    becomes_current = activate and customer_id is not None
    if becomes_current:
        session.execute(
            update(ChurnPrediction)
            .where(
                ChurnPrediction.customer_id == customer_id,
                ChurnPrediction.data_origin == data_origin,
                ChurnPrediction.is_current.is_(True),
            )
            .values(is_current=False)
        )
    prediction = ChurnPrediction(
        customer_id=customer_id,
        batch_id=batch_id,
        idempotency_key=idempotency_key,
        request_payload_sha256=request_payload_sha256,
        model_version=model_version,
        churn_probability=churn_probability,
        decision_threshold=decision_threshold,
        risk_band=risk_band,
        factors=factors,
        feature_snapshot=feature_snapshot,
        feature_schema_version=feature_schema_version,
        feature_schema_digest=feature_schema_digest,
        factors_schema_digest=factors_schema_digest,
        data_origin=data_origin,
        score_at=score_at,
        feature_as_of=feature_as_of,
        outcome_window_start=outcome_window_start,
        outcome_window_end=outcome_window_end,
        is_current=becomes_current,
    )
    session.add(prediction)
    session.commit()
    return prediction


def record_outcome(
    session: Session,
    prediction_id: uuid.UUID,
    *,
    expected_version: int,
    actual_churn: bool,
    origin: DataOrigin,
    recorded_by: str,
    observed_at: datetime | None = None,
) -> ChurnPrediction:
    """Record or correct an outcome with optimistic concurrency (ADR-0014).

    The first record sets ``outcome_recorded_at``; corrections keep it and
    set ``outcome_updated_at``, preserving the administrative timeline
    (ADR-0005/ADR-0006). A stale ``expected_version`` raises
    ``ConcurrencyConflictError`` without touching the stored outcome — note
    the conflict path rolls the session back, discarding any pending work
    the caller had in the same transaction. Database checks still apply —
    e.g. a prospective "did not churn" before window maturity is rejected
    by the maturity constraint. A correction that omits ``observed_at``
    keeps the previously recorded event moment instead of erasing it
    (ADR-0014: corrections never silently rewrite the past).
    """
    now = datetime.now(UTC)
    already_recorded = ChurnPrediction.outcome_recorded_at.is_not(None)
    if observed_at is None:
        # A correction that omits the event moment keeps the stored one.
        observed_value: object = case(
            (already_recorded, ChurnPrediction.outcome_observed_at),
            else_=None,
        )
    else:
        observed_value = observed_at
    statement = (
        update(ChurnPrediction)
        .where(
            ChurnPrediction.id == prediction_id,
            ChurnPrediction.outcome_version == expected_version,
        )
        .values(
            actual_churn=actual_churn,
            outcome_origin=origin,
            outcome_recorded_by=recorded_by,
            outcome_observed_at=observed_value,
            outcome_recorded_at=case(
                (already_recorded, ChurnPrediction.outcome_recorded_at),
                else_=now,
            ),
            outcome_updated_at=case(
                (already_recorded, now),
                else_=None,
            ),
            outcome_version=ChurnPrediction.outcome_version + 1,
        )
        .returning(ChurnPrediction.id)
    )
    updated_id = session.execute(statement).scalar_one_or_none()
    if updated_id is None:
        session.rollback()
        exists = session.scalar(
            select(ChurnPrediction.id).where(ChurnPrediction.id == prediction_id)
        )
        if exists is None:
            raise ValueError(f"unknown prediction: {prediction_id}")
        raise ConcurrencyConflictError(
            f"outcome of {prediction_id} was changed concurrently;"
            f" expected version {expected_version}"
        )
    session.commit()
    # Re-read authoritatively: the bulk UPDATE above does not refresh the
    # identity map (the explicit RETURNING clause carries only the id).
    prediction = session.scalar(
        select(ChurnPrediction)
        .where(ChurnPrediction.id == prediction_id)
        .execution_options(populate_existing=True)
    )
    if prediction is None:  # pragma: no cover - committed row must exist
        raise ValueError(f"unknown prediction: {prediction_id}")
    return prediction
