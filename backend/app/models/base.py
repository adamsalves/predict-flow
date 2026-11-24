"""Declarative base and shared column types for the domain schema (issue #7).

Constraints follow the ADRs: origins are explicit (ADR-0006), outcomes are
born null and only mature inside their observation window (ADR-0005), and
database constraints are the final protection against races (ADR-0014).
"""

from __future__ import annotations

from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase

# JSON payload values are modeled explicitly (no Any): prediction payloads
# are JSON objects whose leaves are scalars or nested containers.
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class DataOrigin(StrEnum):
    """Explicit data origin (ADR-0006): never inferred, always stored."""

    DEMO_BACKFILL = "demo_backfill"
    HISTORICAL_TEST = "historical_test"
    PROSPECTIVE = "prospective"


def data_origin_enum(column: str) -> sa.Enum:
    """VARCHAR-backed enum with a CHECK constraint (no PG enum type).

    A fresh instance per column, named after it, keeps CHECK constraint
    names unique per table (``churn_predictions`` has two origin columns).
    Values (not member names) are stored so SQL filters match ADR-0006
    literals such as 'prospective'.
    """
    return sa.Enum(
        DataOrigin,
        native_enum=False,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
        # SQLAlchemy 1.4+ defaults create_constraint to False: without this
        # the CHECK exists only in Python and the database would accept any
        # string — constraints in the database are the final protection
        # (ADR-0014).
        create_constraint=True,
        name=column,
        length=20,
    )


class Base(DeclarativeBase):
    metadata = sa.MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
