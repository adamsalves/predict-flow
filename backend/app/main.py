from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_error_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.v1.health import router as health_router
from app.api.v1.model import router as model_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.model_loader import load_active_model

setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the active model exactly once per process (issue #8).

    Every gate failure (missing artifact, divergent digest, incompatible
    contract or environment) raises here and aborts startup — the API
    never serves traffic without a verified model (ADR-0003, ADR-0017).
    Settings are re-read inside the lifespan so tests can repoint the
    models volume via environment variables.
    """
    current = get_settings()
    app.state.loaded_model = load_active_model(
        model_path=Path(current.MODEL_PATH),
        model_version=current.MODEL_VERSION,
        registry_path=(
            Path(current.MODEL_REGISTRY_PATH)
            if current.MODEL_REGISTRY_PATH
            else None
        ),
    )
    yield
    app.state.loaded_model = None


app = FastAPI(
    title="predict-flow",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Added last, runs first: every response — including CORS rejections and
# error envelopes — carries the request id and an access log line.
app.add_middleware(RequestContextMiddleware)
register_error_handlers(app)
app.include_router(health_router, prefix="/api/v1")
app.include_router(model_router, prefix="/api/v1")
