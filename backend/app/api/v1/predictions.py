"""Synchronous prediction endpoint (issue #9).

The route stays thin: validation and the envelope come from the
cross-cutting conventions (#40), the model from the lifespan (#8), the
concurrency gate from app state, and orchestration from
``app.services.predict``. Inference runs in the threadpool (``def``,
ADR-0012) and never touches the event loop.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import AfterValidator
from sqlalchemy.orm import Session

from app.api.errors import COMMON_ERROR_RESPONSES
from app.db.session import get_session
from app.models import ChurnPrediction, Customer
from app.schemas.errors import ErrorResponse
from app.schemas.outcome import OutcomePatchRequest
from app.schemas.prediction import (
    OutcomeState,
    PredictionSummary,
    PredictRequest,
    PredictResponse,
)
from app.services.model_loader import LoadedModel
from app.services.outcomes import apply_outcome
from app.services.predict import predict

router = APIRouter(tags=["predict"], responses=COMMON_ERROR_RESPONSES)


def _normalized_key(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("idempotency key must not be blank")
    return normalized


IdempotencyKey = Annotated[
    str,
    Header(min_length=1, description="Client-generated key for safe retries"),
    AfterValidator(_normalized_key),
]


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=201,
    responses={
        200: {"model": PredictResponse, "description": "Idempotent replay"},
        409: {
            "model": ErrorResponse,
            "description": "Key reused with a different payload",
        },
        429: {
            "model": ErrorResponse,
            "description": "Inference concurrency limit reached",
        },
        503: {
            "model": ErrorResponse,
            "description": "Model or concurrency gate not loaded",
        },
    },
)
def create_prediction(
    request: Request,
    response: Response,
    payload: PredictRequest,
    idempotency_key: IdempotencyKey,
) -> PredictResponse:
    """Score one contract-shaped input row, persist and explain it.

    A repeated call with the same key and an equivalent payload replays
    the recorded result (200) without duplicating history; the same key
    with a different payload is a conflict (409). A fresh prediction is
    201.
    """
    loaded: LoadedModel | None = getattr(request.app.state, "loaded_model", None)
    if loaded is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    semaphore = getattr(request.app.state, "predict_semaphore", None)
    if semaphore is None:
        raise HTTPException(status_code=503, detail="predict gate not initialized")
    result, created = predict(
        loaded,
        payload,
        idempotency_key=idempotency_key,
        semaphore=semaphore,
    )
    if not created:
        response.status_code = 200
    return result


def _summary_from_row(session: Session, row: ChurnPrediction) -> PredictionSummary:
    """Project the persisted row (with its customer identity) to the API
    contract shared by the outcome patch and the list endpoint (issue #10)."""
    customer_id_ext: str | None = None
    if row.customer_id is not None:
        customer = session.get_one(Customer, row.customer_id)
        customer_id_ext = customer.customer_id_ext
    return PredictionSummary(
        prediction_id=row.id,
        customer_id_ext=customer_id_ext,
        model_version=row.model_version,
        data_origin=row.data_origin,
        score_at=row.score_at,
        churn_probability=row.churn_probability,
        decision_threshold=row.decision_threshold,
        risk_band=row.risk_band,
        is_current=row.is_current,
        outcome=OutcomeState(
            actual_churn=row.actual_churn,
            outcome_origin=row.outcome_origin,
            outcome_recorded_by=row.outcome_recorded_by,
            outcome_observed_at=row.outcome_observed_at,
            outcome_recorded_at=row.outcome_recorded_at,
            outcome_updated_at=row.outcome_updated_at,
            censored_at=row.censored_at,
            outcome_version=row.outcome_version,
        ),
    )


@router.patch(
    "/predictions/{prediction_id}/outcome",
    response_model=PredictionSummary,
    responses={
        404: {"model": ErrorResponse, "description": "Unknown prediction"},
        409: {
            "model": ErrorResponse,
            "description": (
                "Idempotency key reused with a different payload/target,"
                " stale expected_version, or immature prospective negative"
            ),
        },
    },
)
def update_outcome(
    payload: OutcomePatchRequest,
    prediction_id: uuid.UUID,
    idempotency_key: IdempotencyKey,
    session: Annotated[Session, Depends(get_session)],
) -> PredictionSummary:
    """Record or correct an outcome idempotently (issue #10, ADR-0014).

    A repeated call with the same key, target and equivalent payload
    replays the stored state (200) without re-executing the update; the
    same key against another prediction or with a different payload
    conflicts (409). Domain failures — unknown prediction, stale version,
    immature prospective negative — map to the envelope by the global
    handlers; the route stays thin.
    """
    row, _replay = apply_outcome(
        session,
        prediction_id,
        idempotency_key=idempotency_key,
        expected_version=payload.expected_version,
        actual_churn=payload.actual_churn,
        recorded_by=payload.recorded_by,
        observed_at=payload.observed_at,
    )
    return _summary_from_row(session, row)
