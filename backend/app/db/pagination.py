"""Stable pagination over SQLAlchemy selects (issue #40).

``paginate`` appends a unique tie-break column to the requested ordering,
so the sort is total and pages are deterministic even when the leading
keys repeat; ``total`` is counted over the unpaginated statement.
"""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select
from sqlalchemy.sql.expression import ColumnExpressionArgument

from app.schemas.pagination import PageParams

# Ordering expressions are inherently heterogeneous (ORM attributes,
# columns, unary expressions) and the expression types are invariant in
# their parameter, so no concrete type accepts every column. Any mirrors
# SQLAlchemy's own order_by typing; justified per project convention.
OrderColumn = ColumnExpressionArgument[Any]  # type: ignore[explicit-any]


def paginate[ItemT](
    session: Session,
    stmt: Select[ItemT],
    *,
    order_by: Sequence[OrderColumn],
    tiebreak: OrderColumn,
    params: PageParams,
) -> tuple[list[ItemT], int]:
    """Return one deterministic page plus the total row count.

    ``tiebreak`` (typically the primary key) is always appended last — a
    bare column means ascending — guaranteeing a stable order when the
    ``order_by`` keys tie. ``total`` ignores limit/offset, so clients can
    compute the page count.
    """
    total = session.scalar(select(func.count()).select_from(stmt.subquery()))
    page_stmt = stmt.order_by(*order_by, tiebreak).limit(params.limit).offset(
        params.offset
    )
    items = list(session.scalars(page_stmt).all())
    return items, int(total or 0)
