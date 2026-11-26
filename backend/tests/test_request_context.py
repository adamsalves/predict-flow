"""Request correlation and access logging (issue #40).

The response and the JSON access log share the same request_id; the log
carries method, route template, status and latency — and never payloads,
headers, query strings or feature values.
"""

import json
import logging
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.logging import JsonFormatter
from tests.conftest import build_conventions_app

REQUEST_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _access_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == "predict_flow.access"
    ]


def test_request_id_generated_and_returned(client: TestClient) -> None:
    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200
    assert REQUEST_ID_PATTERN.match(resp.headers["x-request-id"])


def test_request_id_propagated_when_safe(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/health/live", headers={"X-Request-ID": "partner-req_123.abc"}
    )
    assert resp.headers["x-request-id"] == "partner-req_123.abc"


def test_request_id_replaced_when_unsafe(client: TestClient) -> None:
    for unsafe in ("id with spaces", "bad\nid", "bad\tid", "x" * 100):
        resp = client.get("/api/v1/health/live", headers={"X-Request-ID": unsafe})
        assert REQUEST_ID_PATTERN.match(resp.headers["x-request-id"])
        assert resp.headers["x-request-id"] != unsafe


def test_access_log_shares_request_id_with_response(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        resp = client.get("/api/v1/health/live")
    records = _access_records(caplog)
    assert len(records) == 1
    line = json.loads(JsonFormatter().format(records[0]))
    assert line["request_id"] == resp.headers["x-request-id"]
    assert line["method"] == "GET"
    assert line["route"] == "/api/v1/health/live"
    assert line["status"] == 200
    assert line["latency_ms"] >= 0


def test_access_log_uses_route_template_not_raw_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = build_conventions_app()

    @app.get("/items/{item_id}")
    def read_item(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    with caplog.at_level(logging.INFO):
        TestClient(app).get("/items/secret-customer-123")
    (record,) = _access_records(caplog)
    line = json.loads(JsonFormatter().format(record))
    assert line["route"] == "/items/{item_id}"
    assert "secret-customer-123" not in json.dumps(line)


def test_access_log_masks_unmatched_path(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        TestClient(build_conventions_app()).get("/users/john.doe@example.com")
    (record,) = _access_records(caplog)
    line = json.loads(JsonFormatter().format(record))
    assert line["status"] == 404
    assert line["route"] == "<unmatched>"
    assert "john.doe" not in json.dumps(line)


def test_access_log_has_no_payload_headers_or_query(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = build_conventions_app()

    @app.post("/echo")
    def echo(payload: dict[str, object]) -> dict[str, str]:
        return {"status": "ok"}

    with caplog.at_level(logging.INFO):
        resp = TestClient(app).post(
            "/echo?token=tok-query-secret",
            json={"password": "s3cr3t-p4ssw0rd", "features": {"tenure": 12}},
            headers={"Authorization": "Bearer tok-header-secret"},
        )
    assert resp.status_code == 200
    (record,) = _access_records(caplog)
    line = json.dumps(JsonFormatter().format(record))
    for secret in (
        "s3cr3t-p4ssw0rd",
        "tok-query-secret",
        "tok-header-secret",
        "tenure",
        "password",
    ):
        assert secret not in line


def _exploding_app() -> FastAPI:
    app = build_conventions_app()

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("db password is hunter2")

    return app


def test_unexpected_error_returns_generic_envelope(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        resp = TestClient(_exploding_app()).get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["message"] == "internal server error"
    assert body["error"]["request_id"] == resp.headers["x-request-id"]
    # No stack trace, exception message or payload leaks to the client.
    assert "hunter2" not in resp.text
    assert "Traceback" not in resp.text
    assert "RuntimeError" not in resp.text


def test_unexpected_error_keeps_traceback_server_side(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        resp = TestClient(_exploding_app()).get("/boom")
    (record,) = _access_records(caplog)
    line = json.loads(JsonFormatter().format(record))
    assert line["status"] == 500
    assert line["request_id"] == resp.headers["x-request-id"]
    assert "RuntimeError: db password is hunter2" in line["exception"]
