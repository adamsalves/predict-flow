"""Synchronous prediction endpoint (issue #9).

The route stays thin: validation and the envelope come from the
cross-cutting conventions (#40), the model from the lifespan (#8), the
concurrency gate from app state, and orchestration from
``app.services.predict``. Inference runs in the threadpool (``def``,
ADR-0012) and never touches the event loop.
"""

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import AfterValidator

from app.api.errors import COMMON_ERROR_RESPONSES
from app.schemas.errors import ErrorResponse
from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.model_loader import LoadedModel
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
