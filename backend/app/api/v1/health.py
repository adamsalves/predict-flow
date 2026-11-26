from fastapi import APIRouter, HTTPException

from app.api.errors import COMMON_ERROR_RESPONSES
from app.db.session import database_available
from app.schemas.errors import ErrorResponse

router = APIRouter(tags=["health"], responses=COMMON_ERROR_RESPONSES)


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/health/ready",
    responses={503: {"model": ErrorResponse, "description": "Database unavailable"}},
)
def ready() -> dict[str, str]:
    if not database_available():
        raise HTTPException(status_code=503, detail="database unavailable")
    return {"status": "ready"}
