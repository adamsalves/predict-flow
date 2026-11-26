"""Common error envelope for the API (issue #40).

Every error response — validation, domain, HTTP or unexpected — uses the
same minimal DTO, so clients (and the OpenAPI-generated TypeScript types,
ADR-0007) handle exactly one shape. Messages must be client-safe: they
never carry request payloads, feature values, secrets or stack traces.
The ``request_id`` correlates the response with the server-side JSON log
line emitted by the request middleware.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# mypy's disallow_any_explicit flags the inherited BaseModel.__init__(**data: Any)
# at every model definition; the Any lives in the library signature, so no local
# typing change can silence it. Justification recorded per project convention
# (same pattern as app/schemas/manifest.py).


class ErrorDetail(BaseModel):  # type: ignore[explicit-any]
    """One sanitized validation failure: location, message and error type.

    The offending input value and pydantic's context are deliberately
    dropped — echoing them back would leak payload data (PII) into logs
    and error responses.
    """

    model_config = ConfigDict(frozen=True)

    loc: list[str | int]
    msg: str
    type: str


class ErrorBody(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    request_id: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):  # type: ignore[explicit-any]
    model_config = ConfigDict(frozen=True)

    error: ErrorBody
