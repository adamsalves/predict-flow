"""Alembic environment: migrations run against the app metadata (issue #7).

The URL resolution order is: ``sqlalchemy.url`` from the config (used by
tests to target a throwaway database), then the application settings
(``DATABASE_URL``). Credentials are never hardcoded here.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, create_engine, pool

from app.core.config import get_settings
from app.models import Base

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers=False: the default (True) silently disables
    # every logger created before the migration runs — including the API's
    # access log when migrations run in-process (tests, workers).
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return configured
    return get_settings().DATABASE_URL


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_database_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        _run(connection)
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
