"""Migration tests: upgrade builds the schema, downgrade is defined (#7)."""

import sqlalchemy as sa
from alembic import command

from tests.conftest import _alembic_config

DOMAIN_TABLES = {"customers", "customer_services", "churn_predictions"}

EXPECTED_INDEXES = {
    "churn_predictions": {
        "uq_churn_predictions_current",
        "ix_churn_predictions_customer_history",
        "ix_churn_predictions_batch",
        "ix_churn_predictions_origin_score",
        "ix_churn_predictions_worklist",
    },
    "customer_services": {
        "uq_customer_services_current",
        "ix_customer_services_customer_history",
    },
}


def _table_names(url: str) -> set[str]:
    engine = sa.create_engine(url)
    try:
        with engine.connect() as connection:
            return set(sa.inspect(connection).get_table_names())
    finally:
        engine.dispose()


def _index_names(url: str, table: str) -> set[str]:
    engine = sa.create_engine(url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                sa.text("SELECT indexname FROM pg_indexes WHERE tablename = :table"),
                {"table": table},
            )
            return {str(row[0]) for row in rows}
    finally:
        engine.dispose()


def test_upgrade_head_creates_domain_schema(fresh_db_url: str) -> None:
    command.upgrade(_alembic_config(fresh_db_url), "head")
    assert DOMAIN_TABLES <= _table_names(fresh_db_url)
    # Currency and history indexes are verified in the catalog, not assumed.
    for table, expected in EXPECTED_INDEXES.items():
        assert expected <= _index_names(fresh_db_url, table)


def test_downgrade_base_drops_schema_and_upgrade_restores(
    fresh_db_url: str,
) -> None:
    config = _alembic_config(fresh_db_url)
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    # Defined behavior: downgrade removes every domain object; only the
    # alembic bookkeeping table remains.
    assert DOMAIN_TABLES.isdisjoint(_table_names(fresh_db_url))
    command.upgrade(config, "head")
    assert DOMAIN_TABLES <= _table_names(fresh_db_url)
