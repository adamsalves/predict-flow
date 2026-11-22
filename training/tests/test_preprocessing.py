"""Preprocessing pipeline behavior: learned-on-train statistics only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.features import FEATURES, TARGET
from training.preprocessing import (
    PIPELINE_NUMERIC_FEATURES,
    build_pipeline,
)
from training.tests.conftest import FrameFactory

_TOTAL_CHARGES_IDX = PIPELINE_NUMERIC_FEATURES.index("TotalCharges")


def _y(df: pd.DataFrame) -> pd.Series[float]:
    return df[TARGET].map({"No": 0.0, "Yes": 1.0})


def test_imputer_learns_train_median(make_frame: FrameFactory) -> None:
    df = make_frame(30, 10)
    df["TotalCharges"] = np.arange(1.0, 41.0)  # median of 1..40 = 20.5
    pipeline = build_pipeline("lr")
    pipeline.fit(df[[spec.name for spec in FEATURES]], _y(df))

    preprocessor = pipeline.named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["numeric"]
    imputer = numeric.named_steps["imputer"]
    assert imputer.statistics_[_TOTAL_CHARGES_IDX] == pytest.approx(20.5)


def test_transform_uses_train_median_for_new_nan(
    make_frame: FrameFactory,
) -> None:
    df = make_frame(30, 10)
    df["TotalCharges"] = np.arange(1.0, 41.0)
    pipeline = build_pipeline("lr")
    pipeline.fit(df[[spec.name for spec in FEATURES]], _y(df))

    preprocessor = pipeline.named_steps["preprocessor"]
    new = df.head(2).copy()
    new["TotalCharges"] = np.nan
    transformed = np.asarray(
        preprocessor.transform(new[[spec.name for spec in FEATURES]]),
        dtype=np.float64,
    )

    numeric = preprocessor.named_transformers_["numeric"]
    scaler = numeric.named_steps["scaler"]
    restored = (
        transformed[0, _TOTAL_CHARGES_IDX] * scaler.scale_[_TOTAL_CHARGES_IDX]
        + scaler.mean_[_TOTAL_CHARGES_IDX]
    )
    assert restored == pytest.approx(20.5)


def test_unknown_category_does_not_break_transform(
    make_frame: FrameFactory,
) -> None:
    df = make_frame(30, 10)
    pipeline = build_pipeline("lr")
    pipeline.fit(df[[spec.name for spec in FEATURES]], _y(df))

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = [spec.name for spec in FEATURES]
    out_train = preprocessor.transform(df[feature_names])

    unseen = df.head(3).copy()
    unseen["gender"] = "Mystery"
    out_unseen = preprocessor.transform(unseen[feature_names])
    assert out_unseen.shape[1] == out_train.shape[1]


def test_pipeline_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unsupported model"):
        build_pipeline("rf")


def test_pipeline_structure_is_single_sklearn_pipeline(
    make_frame: FrameFactory,
) -> None:
    df = make_frame(30, 10)
    pipeline = build_pipeline("lr")
    pipeline.fit(df[[spec.name for spec in FEATURES]], _y(df))
    steps = [name for name, _ in pipeline.steps]
    assert steps == ["preprocessor", "classifier"]
