"""Synchronous prediction orchestration (issue #9).

The orchestration keeps expensive work outside database sessions (ADR-0014):

1. idempotency lookup in a short session — a recorded key with the same
   payload hash replays the persisted result without re-scoring and
   without a new row; a different hash conflicts (ADR-0014);
2. the concurrency semaphore gates inference/explanation (ADR-0012) —
   when saturated the call fails fast with 429 instead of queueing;
3. scoring and SHAP run with no session open, so no transaction or
   connection is held while the CPU works;
4. persistence happens in one short transaction; the unique index on the
   idempotency key is the final race protection — a concurrent commit with
   the same key resolves to the winner's row (same payload) or a 409
   (different payload), never a duplicate.

A persistence failure after inference never returns an audited success:
the error propagates, nothing is committed, and a retry with the same key
recomputes the (deterministic) result and commits it once.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import session as db_session
from app.models import ChurnPrediction, DataOrigin, JsonValue
from app.schemas.explanation import (
    LocalExplanation,
    factors_schema_digest,
)
from app.schemas.features import FEATURE_SCHEMA_VERSION, feature_schema_digest
from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.customers import register_customer
from app.services.errors import IdempotencyConflictError, InferenceBusyError
from app.services.explain import explain_local
from app.services.model_loader import LoadedModel
from app.services.predictions import find_by_idempotency_key, record_prediction

SessionFactory = Callable[[], Session]


def canonical_payload_hash(request: PredictRequest) -> str:
    """SHA-256 of the canonical payload bound to an idempotency key (ADR-0014).

    Canonicalization is deterministic (sorted keys, compact separators), so
    an equivalent retry hashes identically and a payload with any semantic
    difference does not.
    """
    payload = {
        "customer_id_ext": request.customer_id_ext,
        "features": request.features(),
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _conflict(key: str) -> IdempotencyConflictError:
    return IdempotencyConflictError(
        f"idempotency key {key!r} was already used with a different payload"
    )


def _response_from_row(loaded: LoadedModel, row: ChurnPrediction) -> PredictResponse:
    """Rebuild the API response from the persisted row — replay-stable."""
    explanation = LocalExplanation.model_validate(row.factors)
    band = row.risk_band
    if band == "high":
        # UP037 strips the quotes with ``from __future__ import annotations``,
        # but Literal parameters are values, not annotations — mypy needs them.
        risk_level: Literal["high", "low"] = "high"  # noqa: UP037
    elif band == "low":
        risk_level = "low"
    else:  # pragma: no cover - DB CHECK enforces the band domain
        raise ValueError(f"stored risk band is not servable: {band!r}")
    if row.decision_threshold is None:  # pragma: no cover - DB CHECK enforces
        raise ValueError("stored prediction has no decision threshold")
    return PredictResponse(
        prediction_id=row.id,
        model_version=row.model_version,
        score_at=row.score_at,
        churn_probability=row.churn_probability,
        decision_threshold=row.decision_threshold,
        risk_level=risk_level,
        target_label=loaded.manifest.target.label,
        target_limitations=loaded.manifest.target.limitations,
        explanation=explanation,
    )


def predict(
    loaded: LoadedModel,
    request: PredictRequest,
    *,
    idempotency_key: str,
    semaphore: threading.BoundedSemaphore,
    session_factory: SessionFactory | None = None,
) -> tuple[PredictResponse, bool]:
    """Score one request; returns (response, created).

    ``created`` distinguishes a fresh prediction (201) from an idempotent
    replay or a race resolved to the concurrent winner (200). The payload
    is semantically identical either way — only the status informs the
    client whether this call performed the work.
    """
    factory: SessionFactory = session_factory or db_session.SessionLocal
    payload_hash = canonical_payload_hash(request)

    with factory() as session:
        existing = find_by_idempotency_key(session, idempotency_key)
    if existing is not None:
        if existing.request_payload_sha256 != payload_hash:
            raise _conflict(idempotency_key)
        return _response_from_row(loaded, existing), False

    if not semaphore.acquire(blocking=False):
        raise InferenceBusyError(
            "inference concurrency limit reached; retry the same idempotency key"
        )
    try:
        # No session is open here: inference and SHAP never hold a
        # connection or transaction (ADR-0014).
        features = request.features()
        probability = loaded.predict_proba(features)
        explanation = explain_local(
            loaded.scorer, features, model_version=loaded.model_version
        )
    finally:
        semaphore.release()

    threshold = loaded.manifest.model.threshold.value
    risk_band = "high" if probability >= threshold else "low"
    feature_snapshot: dict[str, JsonValue] = {
        name: value for name, value in features.items()
    }
    with factory() as session:
        customer_id = None
        if request.customer_id_ext is not None:
            customer = register_customer(session, request.customer_id_ext)
            customer_id = customer.id
        try:
            row = record_prediction(
                session,
                customer_id=customer_id,
                batch_id=None,
                model_version=loaded.model_version,
                churn_probability=probability,
                decision_threshold=threshold,
                risk_band=risk_band,
                factors=explanation.model_dump(mode="json"),
                feature_snapshot=feature_snapshot,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                feature_schema_digest=feature_schema_digest(),
                factors_schema_digest=factors_schema_digest(),
                data_origin=DataOrigin.PROSPECTIVE,
                score_at=datetime.now(UTC),
                idempotency_key=idempotency_key,
                request_payload_sha256=payload_hash,
            )
        except IntegrityError:
            session.rollback()
            with factory() as replay:
                winner = find_by_idempotency_key(replay, idempotency_key)
            if winner is None:
                raise
            if winner.request_payload_sha256 != payload_hash:
                raise _conflict(idempotency_key)
            return _response_from_row(loaded, winner), False
    return _response_from_row(loaded, row), True
