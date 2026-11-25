"""Engine and session lifecycle (ADR-0012).

The engine is shared; sessions are never global — one session is created
per request/operation and injected via the ``get_session`` dependency.
Application services delimit transactions; routes stay thin.
"""

import math
from collections.abc import Generator

import psycopg
from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

engine = create_engine(get_settings().DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Generator[Session, None, None]:
    """Yield one session per request/operation and always close it.

    The dependency never commits: transaction boundaries belong to the
    service layer, and closing without commit rolls back open work.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def to_psycopg_dsn(url: str) -> str:
    """Convert a SQLAlchemy URL to a libpq DSN usable by psycopg."""
    return make_url(url).render_as_string(hide_password=False).replace(
        "postgresql+psycopg://", "postgresql://"
    )


def database_available(timeout_seconds: float | None = None) -> bool:
    """Bounded readiness probe (issue #40).

    A fresh connection — not the pool — with an explicit connect timeout
    and statement timeout, so a slow or blackholed database fails the
    check within a bounded time instead of hanging the readiness
    endpoint; worst case is roughly connect timeout plus statement
    timeout. Defaults come from ``DB_READINESS_TIMEOUT_SECONDS``.
    """
    settings = get_settings()
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.DB_READINESS_TIMEOUT_SECONDS
    )
    try:
        with psycopg.connect(
            to_psycopg_dsn(settings.DATABASE_URL),
            connect_timeout=max(1, math.ceil(timeout)),
            options=f"-c statement_timeout={max(1, int(timeout * 1000))}",
        ) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False
