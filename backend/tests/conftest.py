"""Shared fixtures: throwaway databases built by the real migration (issue #7).

Constraint and concurrency behavior is tested against a real PostgreSQL,
never against ORM mocks (ADR-0012) and never via ``create_all``: the schema
under test is the one the Alembic migration produces. Each database is
created fresh and dropped afterwards, so the developer database is never
touched. Without a reachable Postgres every fixture skips, like the health
check test.

Issue #8 adds ``published_models_dir``: two real training runs on synthetic
data whose digests land in the trusted publication registry, so the model
loader tests exercise the actual cross-package contract — training
publishes, serving verifies.
"""

import shutil
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
from app.services.model_loader import REGISTRY_FILENAME
from training.dataset import DatasetProvenance
from training.tests.conftest import synthetic_frame
from training.train import train

BACKEND_DIR = Path(__file__).resolve().parents[1]

SAMPLE_INPUT: dict[str, object] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 1074.0,
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session")
def published_models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two published model versions (v1, v2) trained on synthetic data.

    ``training.train`` writes artifacts, manifests and the trusted
    publication registry exactly as in production, so loader tests verify
    the real registry written by the training package.
    """
    output_dir = tmp_path_factory.mktemp("models")
    df = synthetic_frame(120, 60)
    provenance = DatasetProvenance(
        name="synthetic mini dataset",
        source_url="local://synthetic",
        license="synthetic, no license",
        sha256="0" * 64,
        n_rows=len(df),
        unit="one synthetic record per customer",
    )
    for version, seed in (("v1", 42), ("v2", 43)):
        train(
            df=df,
            provenance=provenance,
            seed=seed,
            version=version,
            output_dir=output_dir,
        )
    return output_dir


@pytest.fixture
def sample_input() -> dict[str, object]:
    return dict(SAMPLE_INPUT)


def stage_versions(
    published: Path,
    dest: Path,
    versions: tuple[str, ...],
    *,
    with_registry: bool = True,
) -> Path:
    """Copy published version directories (and the trusted registry) to dest.

    Tampering tests mutate the copies, never the session-scoped originals.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for version in versions:
        shutil.copytree(published / version, dest / version)
    if with_registry:
        shutil.copy(published / REGISTRY_FILENAME, dest / REGISTRY_FILENAME)
    return dest


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
