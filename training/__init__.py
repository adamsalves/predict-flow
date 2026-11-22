"""Training package: reproducible ingestion, disjoint splits, single sklearn
Pipeline and a complete training manifest for the churn baseline (ADR-0002,
ADR-0004, ADR-0009, ADR-0013, ADR-0017)."""

from training.features import (
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    ID_COLUMN,
    TARGET,
    feature_schema_digest,
)

__all__ = [
    "FEATURES",
    "FEATURE_SCHEMA_VERSION",
    "ID_COLUMN",
    "TARGET",
    "feature_schema_digest",
]
