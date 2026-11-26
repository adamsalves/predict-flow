"""Shared pagination contract (issue #40).

Every list endpoint uses the same convention: bounded ``limit``,
``offset`` and a stable ordering with a unique tie-break, so pages never
repeat or skip rows under concurrent writes. Routers parse params via
``app.api.pagination.page_params`` and services apply them via
``app.db.pagination.paginate`` — no endpoint reimplements this.
"""

from pydantic import BaseModel, ConfigDict, Field

MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20


class PageParams(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    limit: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)


class Page[ItemT](BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    items: list[ItemT]
    total: int
    limit: int
    offset: int
