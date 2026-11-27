"""Outcome patch contract (issue #10, ADR-0005/0006/0014).

One audited outcome record/correction: the verdict, the optional event
moment (the Telco dataset has none), the version the client built the
patch from and the self-declared author (auth is issue #17 — the field is
auditable and takes part in the canonical idempotency hash). The outcome
origin is never accepted from the client: it is derived from the
prediction row (ADR-0006), so no field for it exists here and extra fields
are rejected.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
)

# Author is self-declared for now (auth is #17): trimmed, non-empty,
# bounded — the canonical value is what the audit trail stores and what
# the idempotency hash binds.
_AUTHOR_MAX_LENGTH = 128


def _trim_author(value: str) -> str:
    return value.strip()


def _require_offset(value: datetime | None) -> datetime | None:
    """Reject naive datetimes (issue #10).

    A naive datetime would be normalized from the process timezone and the
    canonical idempotency hash must stay stable across environments; the
    client declares the event moment with an explicit offset.
    """
    if value is not None and value.utcoffset() is None:
        raise ValueError("observed_at must carry an explicit UTC offset")
    return value


class OutcomePatchRequest(BaseModel):  # type: ignore[explicit-any]  # inherited BaseModel.__init__(**data: Any); same justification as PredictRequest
    """The body of one idempotent outcome patch (issue #10, D2)."""

    model_config = ConfigDict(extra="forbid")

    actual_churn: bool
    observed_at: Annotated[
        datetime | None,
        AfterValidator(_require_offset),
        Field(
            description=(
                "Event moment with an explicit UTC offset;"
                " the Telco dataset has none"
            )
        ),
    ] = None
    expected_version: int = Field(
        ge=0,
        description="Outcome version the patch was built from (optimistic concurrency)",
    )
    recorded_by: Annotated[
        str,
        BeforeValidator(_trim_author),
        Field(max_length=_AUTHOR_MAX_LENGTH, min_length=1),
    ]
