"""Error envelope builder and exception handlers (issue #40).

One builder produces every error response so validation, domain, HTTP and
unexpected failures share the envelope defined in ``app.schemas.errors``.
The handlers are registered once on the application; domain routers never
reimplement error mapping.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from app.schemas.errors import ErrorBody, ErrorDetail, ErrorResponse
from app.services.errors import DomainError

# OpenAPI fragment documenting the envelope on every route of a router
# (issue #40): validation and unexpected failures are possible anywhere.
COMMON_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    422: {"model": ErrorResponse, "description": "Request validation failed"},
    500: {"model": ErrorResponse, "description": "Unexpected internal error"},
}


def current_request_id(request: Request) -> str:
    """Read the correlation id stashed by the request middleware.

    An empty string signals broken wiring (middleware missing) — tests
    assert on it, and in the real app the middleware always runs first.
    """
    state = request.scope.get("state")
    request_id = state.get("request_id") if isinstance(state, dict) else None
    return request_id if isinstance(request_id, str) else ""


def error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    details: list[ErrorDetail] | None = None,
) -> JSONResponse:
    envelope = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            request_id=request_id,
            details=details,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(mode="json", exclude_none=True),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Map expected failures to the common envelope (ADR-0012/ADR-0014)."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            ErrorDetail(
                loc=list(error["loc"]),
                msg=error["msg"],
                type=error["type"],
            )
            for error in exc.errors()
        ]
        return error_response(
            status_code=422,
            code="validation_error",
            message="request validation failed",
            request_id=current_request_id(request),
            details=details,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # HTTPException details are authored by our code and client-safe.
        return error_response(
            status_code=exc.status_code,
            code=_http_error_code(exc.status_code),
            message=str(exc.detail),
            request_id=current_request_id(request),
        )

    @app.exception_handler(DomainError)
    async def domain_error_handler(
        request: Request, exc: DomainError
    ) -> JSONResponse:
        # Domain error messages are part of the API contract: services
        # must phrase them client-safe (no payload, features or internals).
        return error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=str(exc),
            request_id=current_request_id(request),
        )


def _http_error_code(status_code: int) -> str:
    from http import HTTPStatus

    try:
        return HTTPStatus(status_code).phrase.lower().replace(" ", "_")
    except ValueError:
        return "http_error"
