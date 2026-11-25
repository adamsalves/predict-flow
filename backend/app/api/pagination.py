"""Pagination dependency for list endpoints (issue #40).

``limit`` is bounded (default 20, max 100) and ``offset`` non-negative;
out-of-range values fail validation and return the common 422 envelope.
"""

from typing import Annotated

from fastapi import Query

from app.schemas.pagination import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, PageParams


def page_params(
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PageParams:
    return PageParams(limit=limit, offset=offset)
