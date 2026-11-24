"""Engine and session lifecycle (ADR-0012).

The engine is shared; sessions are never global — one session is created
per request/operation and injected via the ``get_session`` dependency.
Application services delimit transactions; routes stay thin.
"""

from collections.abc import Generator

from sqlalchemy import create_engine, text
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


def database_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
