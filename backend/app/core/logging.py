"""Minimal JSON logging (issue #40).

Application logs are single-line JSON so the demo's log stream is
greppable and ready for a future collector (#15 adds metrics and SLOs on
top). The access log carries correlation and timing — request_id, method,
route template, status, latency — and never payloads, query strings,
headers or feature values.
"""

import json
import logging
import sys
from datetime import UTC, datetime

# Attributes every LogRecord carries; anything beyond them was attached by
# the caller via ``extra=`` and belongs in the JSON payload.
_STANDARD_ATTRIBUTES = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)


class JsonFormatter(logging.Formatter):
    """Render one log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRIBUTES and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging() -> None:
    """Attach the JSON handler to the root logger, exactly once.

    Non-destructive: handlers installed by the runner (uvicorn, pytest)
    are left in place. App loggers propagate to the root logger, so their
    records come out as JSON lines on stdout.
    """
    root = logging.getLogger()
    if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(logging.INFO)
