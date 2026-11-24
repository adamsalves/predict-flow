"""Domain errors raised by application services.

HTTP mapping arrives with issue #40; until then services raise these
exceptions and tests assert on them directly.
"""


class DomainError(Exception):
    """Base class for domain-level failures."""


class ConcurrencyConflictError(DomainError):
    """An update was based on a stale version (ADR-0014): no lost update."""
