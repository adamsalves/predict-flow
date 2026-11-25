"""Serving/training parity for the versioned contracts (ADR-0013, ADR-0017).

The serving image does not ship the training package, so the feature
contract and the publication-registry format are mirrored in
``app.schemas.features`` and ``app.services.model_loader``. These tests
fail on any drift between the mirrors — a contract change must be a
coordinated release, never a silent divergence.
"""

from __future__ import annotations

import training.features
import training.publication
from app.schemas.features import (
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    feature_schema_digest,
)
from app.services import model_loader


def test_feature_contract_matches_training() -> None:
    assert FEATURE_SCHEMA_VERSION == training.features.FEATURE_SCHEMA_VERSION
    assert feature_schema_digest() == training.features.feature_schema_digest()
    served = [(spec.name, spec.dtype, spec.required) for spec in FEATURES]
    trained = [
        (spec.name, spec.dtype, spec.required) for spec in training.features.FEATURES
    ]
    assert served == trained


def test_registry_format_matches_training() -> None:
    assert model_loader.REGISTRY_FILENAME == training.publication.REGISTRY_FILENAME
    assert (
        model_loader.REGISTRY_SCHEMA_VERSION
        == training.publication.REGISTRY_SCHEMA_VERSION
    )
    assert model_loader.MODEL_FILENAME == training.publication.MODEL_FILENAME
    assert model_loader.MANIFEST_FILENAME == training.publication.MANIFEST_FILENAME
    assert set(model_loader._RECORD_FIELDS) == set(
        training.publication._RECORD_FIELDS
    )
