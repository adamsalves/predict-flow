"""Request correlation, access logging and the last-resort error net.

One pure-ASGI middleware owns the request lifecycle (issue #40):

- generates or propagates a safe ``X-Request-ID`` and stashes it in the
  request state, so the error envelope and the access log share it;
- emits one JSON access log line per request with method, route template,
  status and latency — never payloads, query strings or headers;
- catches anything that escaped the exception handlers and answers with
  the generic 500 envelope: the client gets no stack trace, no exception
  message and no payload echo; the server log keeps the full traceback.
"""

import logging
import re
import time
import uuid
from urllib.parse import quote

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import error_response

access_log = logging.getLogger("predict_flow.access")

REQUEST_ID_HEADER = b"x-request-id"
# Only short, log-safe ids are propagated; anything else (newlines, spaces,
# oversized values) is replaced by a generated id to prevent log injection.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        start = time.perf_counter()
        status = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                headers = message.setdefault("headers", [])
                headers.append((REQUEST_ID_HEADER, request_id.encode("latin-1")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            access_log.exception(
                "unhandled error",
                extra=_log_fields(scope, request_id, status=500, start=start),
            )
            if response_started:
                # Headers already sent: a clean envelope is impossible.
                raise
            response = error_response(
                status_code=500,
                code="internal_error",
                message="internal server error",
                request_id=request_id,
            )
            await response(scope, receive, send_wrapper)
            return
        access_log.info(
            "request",
            extra=_log_fields(scope, request_id, status=status, start=start),
        )


def _incoming_request_id(scope: Scope) -> str | None:
    headers = scope.get("headers")
    if not isinstance(headers, list):
        return None
    for name, value in headers:
        if name == REQUEST_ID_HEADER and isinstance(value, bytes):
            candidate = value.decode("latin-1")
            if _SAFE_REQUEST_ID.match(candidate):
                return candidate
            return None
    return None


def _log_fields(
    scope: Scope, request_id: str, *, status: int, start: float
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "method": scope.get("method", ""),
        "route": _route_template(scope),
        "status": status,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }


def _route_template(scope: Scope) -> str:
    """A low-cardinality, PII-free label for the request path.

    Matched routes: the raw path with every captured path parameter masked
    by its ``{name}`` placeholder — over-masking a coincidental segment is
    cosmetically wrong but never leaks. Unmatched routes (404s) get a fixed
    label: an arbitrary mistyped URL can itself carry PII.
    """
    route = scope.get("route")
    raw_path = scope.get("path", "")
    if not isinstance(raw_path, str):
        return ""
    if route is None:
        return "<unmatched>"
    path_params = scope.get("path_params")
    template = raw_path
    if isinstance(path_params, dict):
        for name, value in path_params.items():
            for form in {str(value), quote(str(value), safe="")}:
                template = template.replace(form, f"{{{name}}}")
    return template
