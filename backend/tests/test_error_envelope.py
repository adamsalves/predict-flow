"""Error contract: validation, domain, HTTP and unexpected failures (issue #40).

Every failure leaves the API in the same envelope, correlated by
request_id; validation details never echo the offending input (PII).
"""

from fastapi import HTTPException, Query
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.services.errors import ConcurrencyConflictError, DomainError
from tests.conftest import build_conventions_app

# See app/schemas/manifest.py for the justified explicit-any ignores.


class _Payload(BaseModel):  # type: ignore[explicit-any]
    tenure: int


def _app() -> TestClient:
    app = build_conventions_app()

    @app.post("/validated")
    def validated(payload: _Payload) -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/limited")
    def limited(limit: int = Query(ge=1)) -> dict[str, int]:
        return {"limit": limit}

    @app.get("/domain-error")
    def domain_error() -> None:
        raise DomainError("customer snapshot is sealed")

    @app.get("/conflict")
    def conflict() -> None:
        raise ConcurrencyConflictError("stale outcome version")

    @app.get("/unavailable")
    def unavailable() -> None:
        raise HTTPException(status_code=503, detail="database unavailable")

    return TestClient(app)


def test_validation_error_envelope_is_sanitized() -> None:
    resp = _app().post("/validated", json={"tenure": "not-a-number-PII"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["request_id"] == resp.headers["x-request-id"]
    (detail,) = body["error"]["details"]
    assert detail["loc"] == ["body", "tenure"]
    assert detail["type"] == "int_parsing"
    assert detail["msg"]
    # The offending value never comes back — not in details, not anywhere.
    assert "not-a-number-PII" not in resp.text
    assert "input" not in detail
    assert "ctx" not in detail


def test_validation_error_on_query_param() -> None:
    resp = _app().get("/limited", params={"limit": 0})
    assert resp.status_code == 422
    (detail,) = resp.json()["error"]["details"]
    assert detail["loc"] == ["query", "limit"]


def test_domain_error_envelope() -> None:
    resp = _app().get("/domain-error")
    assert resp.status_code == 400
    body = resp.json()["error"]
    assert body["code"] == "domain_error"
    assert body["message"] == "customer snapshot is sealed"
    assert body["request_id"] == resp.headers["x-request-id"]


def test_concurrency_conflict_maps_to_409() -> None:
    resp = _app().get("/conflict")
    assert resp.status_code == 409
    body = resp.json()["error"]
    assert body["code"] == "concurrency_conflict"
    assert body["message"] == "stale outcome version"


def test_http_exception_envelope() -> None:
    resp = _app().get("/unavailable")
    assert resp.status_code == 503
    body = resp.json()["error"]
    assert body["code"] == "service_unavailable"
    assert body["message"] == "database unavailable"


def test_unknown_route_uses_envelope() -> None:
    resp = _app().get("/no-such-route")
    assert resp.status_code == 404
    body = resp.json()["error"]
    assert body["code"] == "not_found"
    assert body["request_id"] == resp.headers["x-request-id"]


def _envelope_ref(response: dict[str, object]) -> str:
    content = response["content"]
    assert isinstance(content, dict)
    schema = content["application/json"]["schema"]
    return str(schema["$ref"])


def test_openapi_documents_error_envelope(client: TestClient) -> None:
    """The OpenAPI contract (ADR-0007) shows the envelope on every route."""
    spec = client.get("/api/openapi.json").json()
    assert "ErrorResponse" in spec["components"]["schemas"]
    paths = spec["paths"]
    live = paths["/api/v1/health/live"]["get"]["responses"]
    assert set(live) >= {"200", "422", "500"}
    ready = paths["/api/v1/health/ready"]["get"]["responses"]
    assert set(ready) >= {"200", "422", "500", "503"}
    for response in (live["422"], live["500"], ready["503"]):
        assert _envelope_ref(response).endswith("/ErrorResponse")
    info = paths["/api/v1/model/info"]["get"]["responses"]
    assert _envelope_ref(info["503"]).endswith("/ErrorResponse")
