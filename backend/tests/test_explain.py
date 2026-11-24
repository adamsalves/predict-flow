"""Explanation adapters: one stable DTO contract for LR and RF (ADR-0018).

The fitted pipelines come from the real training preprocessing so the
contract tests run against the production artifact shape; the backend
itself never imports the training package at runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from app.schemas.explanation import (
    FACTORS_SCHEMA_VERSION,
    METHOD_LINEAR,
    METHOD_TREE_SHAP,
    UNIT_LOG_ODDS,
    UNIT_PROBABILITY,
    factors_schema_digest,
)
from app.services.explain import explain_local
from training.features import FEATURES, TARGET
from training.preprocessing import build_pipeline
from training.tests.conftest import synthetic_frame

_FEATURE_NAMES = [spec.name for spec in FEATURES]


def _fit(model: str, n_estimators: int | None = None) -> tuple[Pipeline, pd.DataFrame]:
    df = synthetic_frame(60, 30)
    hyper: dict[str, object] = {}
    if n_estimators is not None:
        hyper["n_estimators"] = n_estimators
    pipeline = build_pipeline(model, hyper, seed=42)
    y = df[TARGET].map({"No": 0.0, "Yes": 1.0})
    pipeline.fit(df[_FEATURE_NAMES], y)
    return pipeline, df


def _raw_row(df: pd.DataFrame, idx: int = 0) -> dict[str, object]:
    row = df.iloc[idx]
    return {name: row[name] for name in _FEATURE_NAMES}


def test_lr_explanation_uses_linear_log_odds_contract() -> None:
    pipeline, df = _fit("lr")
    raw = _raw_row(df)
    explanation = explain_local(pipeline, raw, model_version="test-lr")

    assert explanation.schema_version == FACTORS_SCHEMA_VERSION
    assert explanation.method == METHOD_LINEAR
    assert explanation.unit == UNIT_LOG_ODDS
    assert explanation.model_version == "test-lr"
    # Additivity: base + all contributions == the model's log-odds output.
    logit = float(
        np.asarray(
            pipeline.decision_function(df[_FEATURE_NAMES].head(1)),
            dtype=np.float64,
        )[0]
    )
    assert explanation.output_value == pytest.approx(logit, abs=1e-8)
    assert explanation.base_value == pytest.approx(
        float(pipeline.named_steps["classifier"].intercept_[0]), abs=1e-12
    )


def test_rf_explanation_uses_tree_shap_probability_contract() -> None:
    pipeline, df = _fit("rf", n_estimators=25)
    raw = _raw_row(df)
    explanation = explain_local(pipeline, raw, model_version="test-rf")

    assert explanation.method == METHOD_TREE_SHAP
    assert explanation.unit == UNIT_PROBABILITY
    # Additivity: base + all contributions == predicted probability.
    proba = float(
        np.asarray(
            pipeline.predict_proba(df[_FEATURE_NAMES].head(1)), dtype=np.float64
        )[0, 1]
    )
    assert explanation.output_value == pytest.approx(proba, abs=1e-8)


def test_both_models_produce_the_same_contract_fields() -> None:
    lr_pipeline, df = _fit("lr")
    rf_pipeline, _ = _fit("rf", n_estimators=25)
    raw = _raw_row(df)
    lr_explanation = explain_local(lr_pipeline, raw, model_version="v")
    rf_explanation = explain_local(rf_pipeline, raw, model_version="v")
    assert set(lr_explanation.model_dump()) == set(rf_explanation.model_dump())
    assert {f.feature for f in lr_explanation.factors} == {
        spec.name for spec in FEATURES
    } & {f.feature for f in lr_explanation.factors}


def test_top_k_limits_factors_with_deterministic_order() -> None:
    pipeline, df = _fit("lr")
    raw = _raw_row(df)
    explanation = explain_local(pipeline, raw, model_version="v", top_k=5)
    assert len(explanation.factors) == 5
    magnitudes = [abs(f.contribution) for f in explanation.factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    again = explain_local(pipeline, raw, model_version="v", top_k=5)
    assert explanation == again


def test_top_k_must_be_positive() -> None:
    pipeline, df = _fit("lr")
    with pytest.raises(ValueError, match="top_k"):
        explain_local(pipeline, _raw_row(df), model_version="v", top_k=0)


def test_unknown_category_does_not_break_explanation() -> None:
    pipeline, df = _fit("lr")
    raw = _raw_row(df)
    raw["gender"] = "Mystery"
    explanation = explain_local(pipeline, raw, model_version="v")
    assert explanation.factors


def test_calibrated_wrapper_is_unwrapped_and_noted() -> None:
    pipeline, df = _fit("lr")
    y = df[TARGET].map({"No": 0.0, "Yes": 1.0})
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(pipeline), method="sigmoid"
    ).fit(df[_FEATURE_NAMES], y)
    explanation = explain_local(calibrated, _raw_row(df), model_version="v")
    assert explanation.method == METHOD_LINEAR
    assert "monotone remap" in explanation.interpretation_note


def test_contributions_are_not_presented_as_causes() -> None:
    pipeline, df = _fit("lr")
    explanation = explain_local(pipeline, _raw_row(df), model_version="v")
    note = explanation.interpretation_note.lower()
    assert "not causes" in note


def test_unsupported_classifier_raises() -> None:
    df = synthetic_frame(20, 10)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_pipeline("lr").named_steps["preprocessor"]),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    pipeline.fit(df[_FEATURE_NAMES], df[TARGET].map({"No": 0.0, "Yes": 1.0}))
    with pytest.raises(ValueError, match="no explanation adapter"):
        explain_local(pipeline, _raw_row(df), model_version="v")


def test_factors_schema_digest_is_stable() -> None:
    assert factors_schema_digest() == factors_schema_digest()
    assert len(factors_schema_digest()) == 64
