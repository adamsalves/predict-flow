"""Versioned feature contract shared by training and serving (ADR-0013).

The manifest records ``feature_schema_version`` and a SHA-256 computed over
the canonical representation of the features: name, dtype, required flag and
exact order. Serving validates incoming data against the same contract, so
the digest is part of the activation gate, not documentation.

Canonical representation: a JSON list of objects with keys in fixed
alphabetical order (``dtype``, ``name``, ``required``), serialized with
``separators=(",", ":")`` and hashed with SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, NamedTuple

FEATURE_SCHEMA_VERSION = "1"
TARGET = "Churn"
ID_COLUMN = "customerID"

DTypeName = Literal["int", "float", "str"]


class FeatureSpec(NamedTuple):
    name: str
    dtype: DTypeName
    required: bool


# Exact contract order: the manifest records it and serving validates
# predictions and requests against it.
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

NUMERIC_FEATURES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURES if spec.dtype in ("int", "float")
)
CATEGORICAL_FEATURES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURES if spec.dtype == "str"
)


def feature_schema_digest() -> str:
    """SHA-256 of the canonical feature representation (order-sensitive)."""
    canonical = [
        {"dtype": spec.dtype, "name": spec.name, "required": spec.required}
        for spec in FEATURES
    ]
    payload = json.dumps(canonical, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
