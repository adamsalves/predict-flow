"""Training-side access to the factors schema descriptor (ADR-0018).

The single source of truth is ``backend/app/schemas/factors_schema.json``,
which ships inside the serving image; training reads the same file from
the repo checkout, so the manifest records exactly the version and digest
the backend will serve. The explanation method/unit per model kind is
mirrored here for the manifest and kept in parity with the adapters by
``backend/tests/test_explain.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "backend"
    / "app"
    / "schemas"
    / "factors_schema.json"
)

# Explanation method and contribution unit produced by the adapter for each
# model kind (ADR-0018); parity with app.services.explain is test-enforced.
EXPLANATION_BY_MODEL: dict[str, dict[str, str]] = {
    "lr": {"method": "linear", "unit": "log_odds"},
    "rf": {"method": "tree_shap", "unit": "probability"},
}


def _descriptor() -> dict[str, object]:
    descriptor: dict[str, object] = json.loads(
        _SCHEMA_PATH.read_text(encoding="utf-8")
    )
    return descriptor


def factors_schema_version() -> str:
    return str(_descriptor()["schema_version"])


def top_k_default() -> int:
    return int(str(_descriptor()["top_k_default"]))


def factors_schema_digest() -> str:
    payload = json.dumps(_descriptor(), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
