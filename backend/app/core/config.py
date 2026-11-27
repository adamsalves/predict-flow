from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# mypy's disallow_any_explicit flags the inherited BaseSettings.__init__(**data: Any)
# at the class definition; the Any lives in the library signature, so no local
# typing change can silence it. Justification recorded per project convention.
class Settings(BaseSettings):  # type: ignore[explicit-any]
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg://predict:predict@localhost:5432/predict_flow"
    # Bounded database readiness probe (issue #40): connect and statement
    # timeouts derive from this budget, so a slow database cannot hang
    # /health/ready.
    DB_READINESS_TIMEOUT_SECONDS: float = Field(default=2.0, gt=0)
    # Inference/explanation concurrency ceiling (issue #9, ADR-0012):
    # excess requests fail fast with 429 instead of queueing in the
    # threadpool.
    PREDICT_MAX_CONCURRENCY: int = Field(default=2, ge=1)
    # Simulated outcome window for prospective predictions (issue #10,
    # ADR-0005/0009): the API stamps outcome_window_start/end relative to
    # score_at and never accepts a window from the client. A prospective
    # "did not churn" is only accepted after this window has matured.
    OUTCOME_WINDOW_DAYS: int = Field(default=90, gt=0)
    MODEL_PATH: str = "/models"
    # Active model version (ADR-0003): activation and rollback are config
    # changes plus a process restart, never runtime hot-swaps.
    MODEL_VERSION: str = "v1"
    # Trusted digest source (ADR-0017); defaults to the publication
    # registry at the root of the models volume.
    MODEL_REGISTRY_PATH: str | None = None
    CORS_ORIGINS: list[str] = ["http://localhost:4200"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
