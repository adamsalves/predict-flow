"""Serving/training parity for the versioned feature contract (ADR-0013).

The serving image does not ship the training package, so the feature
contract is mirrored in ``app.schemas.features``. This test fails on any
drift between the mirrors — a contract change must be a coordinated
release, never a silent divergence.
"""

from __future__ import annotations

import training.features
from app.schemas.features import (
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    feature_schema_digest,
)


def test_feature_contract_matches_training() -> None:
    assert FEATURE_SCHEMA_VERSION == training.features.FEATURE_SCHEMA_VERSION
    assert feature_schema_digest() == training.features.feature_schema_digest()
    served = [(spec.name, spec.dtype, spec.required) for spec in FEATURES]
    trained = [
        (spec.name, spec.dtype, spec.required) for spec in training.features.FEATURES
    ]
    assert served == trained
