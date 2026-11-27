"""Outcome application with key-based idempotency (issue #10, ADR-0014).

One idempotency key binds one canonical payload to one recorded result on
one prediction — the operation is (key, prediction, payload), so the
target id takes part in the canonical hash: a retry against another
prediction conflicts instead of answering a state it never produced. An
equivalent retry replays the stored state without re-executing the update,
and a concurrent first write of the registry row is arbitrated by the
database's unique constraint — different targets hash differently, so the
loser of the race always conflicts.

The outcome origin is always derived from the prediction row (ADR-0006),
never accepted from the client; the maturity CHECK for prospective
negatives (ADR-0005) is translated into a domain conflict instead of
escaping as an integrity failure.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.constraints import violated_constraint
from app.models import ChurnPrediction
from app.models.outcome_idempotency import OutcomeIdempotency
from app.services.errors import (
    IdempotencyConflictError,
    OutcomeWindowNotMatureError,
    PredictionNotFoundError,
)
from app.services.predictions import record_outcome

# CHECK that rejects a prospective "did not churn" before the outcome
# window has matured (ADR-0005); the driver reports the violated name.
_MATURITY_CHECK = "ck_churn_predictions_prospective_negative_maturity"
# Explicit unique constraint arbitrating the concurrent first write of the
# registry row (ADR-0014).
_KEY_UNIQUE = "uq_outcome_idempotency_key"

# Versioned canonical form (issue #10): changing the hash shape — fields,
# encoding, normalization — requires bumping this version, otherwise
# retries already in flight across the change would spuriously conflict.
CANONICAL_HASH_VERSION = 1


def canonical_payload_hash(
    *,
    prediction_id: uuid.UUID,
    actual_churn: bool,
    observed_at: datetime | None,
    expected_version: int,
    recorded_by: str,
) -> str:
    """SHA-256 of the canonical outcome payload bound to the key (ADR-0014).

    The operation is (key, prediction, payload): the prediction id is part
    of the material. Canonicalization is deterministic (sorted keys, compact
    separators, datetimes in ISO format normalized to UTC), so an equivalent
    retry hashes identically and any semantic difference — including the
    target prediction or the presence of the event moment — does not.
    """
    payload = {
        "actual_churn": actual_churn,
        "expected_version": expected_version,
        "hash_version": CANONICAL_HASH_VERSION,
        "observed_at": (
            None
            if observed_at is None
            else observed_at.astimezone(UTC).isoformat()
        ),
        "prediction_id": str(prediction_id),
        "recorded_by": recorded_by,
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _conflict(key: str) -> IdempotencyConflictError:
    return IdempotencyConflictError(
        f"idempotency key {key!r} was already used with a different payload"
    )


def _current_state(session: Session, prediction_id: uuid.UUID) -> ChurnPrediction:
    """Re-read the stored prediction authoritatively (fresh stored state)."""
    row = session.scalar(
        select(ChurnPrediction)
        .where(ChurnPrediction.id == prediction_id)
        .execution_options(populate_existing=True)
    )
    if row is None:  # pragma: no cover - the FK guarantees the row exists
        raise PredictionNotFoundError(f"prediction {prediction_id} was not found")
    return row


def apply_outcome(
    session: Session,
    prediction_id: uuid.UUID,
    *,
    idempotency_key: str,
    expected_version: int,
    actual_churn: bool,
    recorded_by: str,
    observed_at: datetime | None = None,
) -> tuple[ChurnPrediction, bool]:
    """Record or correct an outcome idempotently; returns (row, replay).

    ``replay`` distinguishes the call that performed the work (``False``)
    from a retry answered with the stored result (``True``). A retry is
    bound to the same prediction: the same key against another prediction
    conflicts even with an otherwise equivalent payload (the target takes
    part in the canonical hash). A stale ``expected_version`` raises
    ``ConcurrencyConflictError``; a prospective negative before window
    maturity raises ``OutcomeWindowNotMatureError``; an unknown prediction
    raises ``PredictionNotFoundError``. A registry miss whose insert loses
    the unique race conflicts as well — different targets hash differently,
    so the race loser cannot be a replay of the winner.
    """
    payload_hash = canonical_payload_hash(
        prediction_id=prediction_id,
        actual_churn=actual_churn,
        observed_at=observed_at,
        expected_version=expected_version,
        recorded_by=recorded_by,
    )
    existing = session.scalar(
        select(OutcomeIdempotency).where(
            OutcomeIdempotency.idempotency_key == idempotency_key
        )
    )
    if existing is not None:
        if existing.payload_sha256 != payload_hash:
            raise _conflict(idempotency_key)
        # Replay: the recorded result is the current stored state; the
        # outcome update is never executed again.
        return _current_state(session, existing.prediction_id), True

    # The origin is derived from the prediction (ADR-0006), never from the
    # client; the CHECK outcome_origin_matches_data_origin is the final
    # protection.
    prediction = session.scalar(
        select(ChurnPrediction).where(ChurnPrediction.id == prediction_id)
    )
    if prediction is None:
        raise PredictionNotFoundError(f"prediction {prediction_id} was not found")

    try:
        updated = record_outcome(
            session,
            prediction_id,
            expected_version=expected_version,
            actual_churn=actual_churn,
            origin=prediction.data_origin,
            recorded_by=recorded_by,
            observed_at=observed_at,
        )
    except IntegrityError as exc:
        session.rollback()
        if violated_constraint(exc) == _MATURITY_CHECK:
            raise OutcomeWindowNotMatureError(
                "prospective negative outcome requires a mature outcome window"
            ) from exc
        raise

    record = OutcomeIdempotency(
        idempotency_key=idempotency_key,
        prediction_id=prediction_id,
        payload_sha256=payload_hash,
        resulted_version=updated.outcome_version,
    )
    session.add(record)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        if violated_constraint(exc) != _KEY_UNIQUE:
            raise
        # With the target in the canonical hash, an insert race can only
        # happen between different predictions, whose hashes differ (the
        # same-prediction race is arbitrated earlier by record_outcome's
        # optimistic concurrency): the loser of the race always conflicts.
        raise _conflict(idempotency_key) from exc
    return updated, False
