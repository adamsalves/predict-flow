"""IntegrityError interpretation helpers (issue #10).

The database, not application locking, is the final arbiter of races and
domain rules (ADR-0014): services translate the constraint reported by the
driver into domain outcomes instead of parsing error message strings.
"""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError


def violated_constraint(exc: IntegrityError) -> str | None:
    """Constraint name reported by the driver, when available."""
    diag = getattr(exc.orig, "diag", None)
    name: str | None = getattr(diag, "constraint_name", None)
    return name
