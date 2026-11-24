"""Shared fixtures: throwaway databases built by the real migration (issue #7).

Constraint and concurrency behavior is tested against a real PostgreSQL,
never against ORM mocks (ADR-0012) and never via ``create_all``: the schema
under test is the one the Alembic migration produces. Each database is
created fresh and dropped afterwards, so the developer database is never
touched. Without a reachable Postgres every fixture skips, like the health
check test.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import database_available
from app.main import app

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _alembic_config(url: str) -> AlembicConfig:
    config = AlembicConfig(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def _admin_dsn() -> str:
    url = make_url(get_settings().DATABASE_URL).set(database="postgres")
    return url.render_as_string(hide_password=False).replace(
        "postgresql+psycopg://", "postgresql://"
    )


def _create_database() -> tuple[str, str]:
    name = f"pf_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(_admin_dsn(), autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    url = (
        make_url(get_settings().DATABASE_URL)
        .set(database=name)
        .render_as_string(hide_password=False)
    )
    return name, url


def _drop_database(name: str) -> None:
    with psycopg.connect(_admin_dsn(), autocommit=True) as conn:
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity"
            " WHERE datname = %s AND pid <> pg_backend_pid()",
            (name,),
        )
        conn.execute(f'DROP DATABASE "{name}"')


def _require_postgres() -> None:
    if not database_available():
        pytest.skip("Postgres indisponível neste ambiente")


@pytest.fixture
def fresh_db_url() -> Iterator[str]:
    """A clean database without any schema (for migration tests)."""
    _require_postgres()
    name, url = _create_database()
    try:
        yield url
    finally:
        _drop_database(name)


@pytest.fixture(scope="session")
def migrated_db_url() -> Iterator[str]:
    """A session-wide database whose schema was built by the migration."""
    _require_postgres()
    name, url = _create_database()
    command.upgrade(_alembic_config(url), "head")
    try:
        yield url
    finally:
        _drop_database(name)


@pytest.fixture
def db_session(migrated_db_url: str) -> Iterator[Session]:
    """A session inside an outer transaction rolled back after each test.

    ``create_savepoint`` makes service-level ``commit()`` release savepoints
    only and ``rollback()`` undo just the current unit of work, so every
    test leaves the shared database untouched. (The default
    ``conditional_savepoint`` would join the outer transaction directly and
    a service rollback would destroy it.)
    """
    engine = create_engine(migrated_db_url)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
        engine.dispose()
