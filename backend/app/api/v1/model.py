"""Model introspection endpoint (issue #8)."""

from fastapi import APIRouter, HTTPException, Request

from app.schemas.model_info import ModelInfo
from app.services.model_loader import LoadedModel

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    """Report the active model version, provenance and headline metrics.

    The model is loaded once during the lifespan; a missing entry means
    the app started without the lifespan (tests) — never a silent 200.
    """
    loaded: LoadedModel | None = getattr(request.app.state, "loaded_model", None)
    if loaded is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return loaded.to_info()
