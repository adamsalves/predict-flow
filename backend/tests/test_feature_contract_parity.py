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
from app.schemas.prediction import PredictRequest
from app.services import model_loader

_DTYPE_NAME: dict[type[object], str] = {str: "str", int: "int", float: "float"}


def test_feature_contract_matches_training() -> None:
    assert FEATURE_SCHEMA_VERSION == training.features.FEATURE_SCHEMA_VERSION
    assert feature_schema_digest() == training.features.feature_schema_digest()
    served = [(spec.name, spec.dtype, spec.required) for spec in FEATURES]
    trained = [
        (spec.name, spec.dtype, spec.required) for spec in training.features.FEATURES
    ]
    assert served == trained


def test_predict_request_schema_matches_feature_contract() -> None:
    """The static API schema mirrors the serving contract (issue #9).

    A field present here but absent from the contract would be silently
    ignored by inference yet enter the idempotency hash and the audit
    snapshot — the drift must fail loudly instead (ADR-0013).
    """
    fields = {
        name: field
        for name, field in PredictRequest.model_fields.items()
        if name != "customer_id_ext"
    }
    assert list(fields) == [spec.name for spec in FEATURES]
    for spec in FEATURES:
        annotation = fields[spec.name].annotation
        assert annotation in _DTYPE_NAME, f"{spec.name}: unexpected {annotation}"
        assert _DTYPE_NAME[annotation] == spec.dtype
        assert fields[spec.name].is_required() is spec.required


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
