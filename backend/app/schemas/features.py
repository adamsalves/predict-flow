"""Static serving-side feature contract (ADR-0013).

The API owns the contract it can serve: version, digest and the exact
feature list. ``ModelLoader`` compares the manifest's recorded contract
against this module before ``joblib.load`` — a model trained against a
different contract never activates. The training side keeps its own copy
in ``training/features.py``; the two are deliberately independent (the
serving image does not ship the training package) and their parity is
enforced by ``backend/tests/test_feature_contract_parity.py``, so a
contract change is a coordinated release, never a silent drift.

Canonical representation (same algorithm as training): a JSON list of
objects with keys in fixed alphabetical order (``dtype``, ``name``,
``required``), serialized with ``separators=(",", ":")`` and hashed with
SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, NamedTuple

FEATURE_SCHEMA_VERSION = "1"

DTypeName = Literal["int", "float", "str"]


class FeatureSpec(NamedTuple):
    name: str
    dtype: DTypeName
    required: bool


# Exact contract order: predictions and explanations are validated against
# this order and the digest recorded in the model manifest.
FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("gender", "str", True),
    FeatureSpec("SeniorCitizen", "int", True),
    FeatureSpec("Partner", "str", True),
    FeatureSpec("Dependents", "str", True),
    FeatureSpec("tenure", "int", True),
    FeatureSpec("PhoneService", "str", True),
    FeatureSpec("MultipleLines", "str", True),
    FeatureSpec("InternetService", "str", True),
    FeatureSpec("OnlineSecurity", "str", True),
    FeatureSpec("OnlineBackup", "str", True),
    FeatureSpec("DeviceProtection", "str", True),
    FeatureSpec("TechSupport", "str", True),
    FeatureSpec("StreamingTV", "str", True),
    FeatureSpec("StreamingMovies", "str", True),
    FeatureSpec("Contract", "str", True),
    FeatureSpec("PaperlessBilling", "str", True),
    FeatureSpec("PaymentMethod", "str", True),
    FeatureSpec("MonthlyCharges", "float", True),
    FeatureSpec("TotalCharges", "float", True),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURES)


def feature_schema_digest() -> str:
    """SHA-256 of the canonical feature representation (order-sensitive).

    Keys are sorted at serialization time so the digest depends only on the
    contract content, never on the literal order of dict keys in source.
    """
    canonical = [
        {"dtype": spec.dtype, "name": spec.name, "required": spec.required}
        for spec in FEATURES
    ]
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
