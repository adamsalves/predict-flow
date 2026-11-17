from fastapi import APIRouter, HTTPException

from app.db.session import database_available

router = APIRouter(tags=["health"])


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def ready() -> dict[str, str]:
    if not database_available():
        raise HTTPException(status_code=503, detail="database unavailable")
    return {"status": "ready"}
