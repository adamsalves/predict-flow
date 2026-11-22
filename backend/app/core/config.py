from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# mypy's disallow_any_explicit flags the inherited BaseSettings.__init__(**data: Any)
# at the class definition; the Any lives in the library signature, so no local
# typing change can silence it. Justification recorded per project convention.
class Settings(BaseSettings):  # type: ignore[explicit-any]
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg://predict:predict@localhost:5432/predict_flow"
    MODEL_PATH: str = "/models"
    CORS_ORIGINS: list[str] = ["http://localhost:4200"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
