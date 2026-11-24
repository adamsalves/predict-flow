"""Calibration/threshold selection on validation, never on test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.calibration import (
    CALIBRATION_METHODS,
    evaluate_at_threshold,
    fit_candidate_calibrators,
    select_calibration,
    select_threshold,
)
from training.features import FEATURES, TARGET
from training.preprocessing import build_pipeline
from training.splits import split_dataset, split_train_for_calibration
from training.tests.conftest import FrameFactory


def _probs(values: list[float]) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    return np.asarray(values, dtype=np.float64)


def test_select_calibration_picks_lowest_brier() -> None:
    y = _probs([0.0, 1.0, 1.0, 0.0])
    choice = select_calibration(
        {
            "none": _probs([0.1, 0.9, 0.8, 0.2]),
            "sigmoid": _probs([0.4, 0.6, 0.6, 0.4]),
            "isotonic": _probs([0.0, 1.0, 1.0, 0.0]),
        },
        y,
    )
    assert choice.method == "isotonic"
    assert set(choice.val_brier_by_method) == set(CALIBRATION_METHODS)


def test_select_calibration_prefers_simpler_method_on_tie() -> None:
    y = _probs([0.0, 1.0])
    same = _probs([0.2, 0.8])
    choice = select_calibration(
        {"isotonic": same, "sigmoid": same, "none": same},
        y,
    )
    assert choice.method == "none"


def test_select_calibration_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown calibration"):
        select_calibration({"platt": _probs([0.5])}, _probs([1.0]))


def test_fitted_calibrators_score_validation_rows(
    make_frame: FrameFactory,
) -> None:
    df = make_frame(120, 60)
    splits = split_dataset(df, seed=42)
    cal_split = split_train_for_calibration(splits.train, seed=42)
    feature_names = [spec.name for spec in FEATURES]

    def y(frame: pd.DataFrame) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return np.asarray(
            frame[TARGET].map({"No": 0.0, "Yes": 1.0}).to_numpy(dtype=np.float64),
            dtype=np.float64,
        )

    pipeline = build_pipeline("lr")
    pipeline.fit(cal_split.fit[feature_names], y(cal_split.fit))

    fitted = fit_candidate_calibrators(
        pipeline, cal_split.calibration[feature_names], y(cal_split.calibration)
    )
    assert set(fitted) == set(CALIBRATION_METHODS)
    for scorer in fitted.values():
        proba = np.asarray(
            scorer.predict_proba(splits.val[feature_names].head(5)),
            dtype=np.float64,
        )
        assert proba.shape == (5, 2)
        assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_select_threshold_maximizes_f1() -> None:
    y = _probs([0.0, 0.0, 1.0, 1.0])
    choice = select_threshold(y, _probs([0.1, 0.4, 0.6, 0.9]))
    assert 0.4 < choice.value <= 0.6
    assert choice.val_f1 == pytest.approx(1.0)
    assert choice.rule == "f1_max_on_validation"


def test_select_threshold_is_deterministic() -> None:
    y = _probs([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    probs = _probs([0.2, 0.8, 0.5, 0.6, 0.7, 0.4])
    first = select_threshold(y, probs)
    second = select_threshold(y, probs)
    assert first == second


def test_evaluate_at_threshold_counts_flagged() -> None:
    y = _probs([0.0, 1.0, 1.0, 0.0])
    metrics = evaluate_at_threshold(y, _probs([0.1, 0.7, 0.6, 0.4]), 0.5)
    assert metrics.n_flagged == 2
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
