"""Domain errors raised by application services.

The HTTP mapping lives in ``app.api.errors`` (issue #40): each class
declares the status code and the stable machine-readable code of the
envelope. Messages must be client-safe — they travel in the response body
and therefore never carry payloads, feature values or internals.
"""


class DomainError(Exception):
    """Base class for domain-level failures."""

    status_code: int = 400
    code: str = "domain_error"


class ConcurrencyConflictError(DomainError):
    """An update was based on a stale version (ADR-0014): no lost update."""

    status_code = 409
    code = "concurrency_conflict"


class IdempotencyConflictError(DomainError):
    """An idempotency key was reused with a different payload (ADR-0014)."""

    status_code = 409
    code = "idempotency_conflict"


class OutcomeWindowNotMatureError(DomainError):
    """A prospective negative outcome was asserted before the outcome
    window matured (ADR-0005/0009, issue #10): censoring and incomplete
    follow-up stay distinct from a confirmed negative."""

    status_code = 409
    code = "outcome_window_not_mature"


class PredictionNotFoundError(DomainError):
    """No prediction exists for the requested identifier (issue #10)."""

    status_code = 404
    code = "prediction_not_found"


class InferenceBusyError(DomainError):
    """The inference concurrency limit is saturated (ADR-0012).

    The client may retry later; the limit protects threadpool and latency
    instead of queueing unbounded work.
    """

    status_code = 429
    code = "too_many_requests"
