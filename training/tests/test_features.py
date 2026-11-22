"""Feature contract digest tests (ADR-0013)."""

from __future__ import annotations

import hashlib
import json

from training.features import FEATURES, feature_schema_digest


def _digest_of(items: list[dict[str, object]]) -> str:
    canonical = [
        {"dtype": item["dtype"], "name": item["name"], "required": item["required"]}
        for item in items
    ]
    payload = json.dumps(canonical, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_digest_is_stable() -> None:
    assert feature_schema_digest() == feature_schema_digest()


def test_digest_matches_manual_recompute() -> None:
    items = [
        {"name": spec.name, "dtype": spec.dtype, "required": spec.required}
        for spec in FEATURES
    ]
    assert feature_schema_digest() == _digest_of(items)


def test_digest_is_order_sensitive() -> None:
    items = [
        {"name": spec.name, "dtype": spec.dtype, "required": spec.required}
        for spec in FEATURES
    ]
    reordered = [items[-1], *items[:-1]]
    assert _digest_of(reordered) != feature_schema_digest()


def test_contract_covers_telco_features() -> None:
    names = [spec.name for spec in FEATURES]
    assert len(names) == 19
    assert len(set(names)) == 19
    dtypes = {spec.name: spec.dtype for spec in FEATURES}
    assert dtypes["tenure"] == "int"
    assert dtypes["SeniorCitizen"] == "int"
    assert dtypes["MonthlyCharges"] == "float"
    assert dtypes["TotalCharges"] == "float"
    assert all(spec.required for spec in FEATURES)
