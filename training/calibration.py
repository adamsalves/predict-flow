"""Calibration and decision threshold, selected on the validation split.

Calibrators are ``CalibratedClassifierCV`` wrappers around a
``FrozenEstimator`` (the modern prefit API), fitted on the calibration
sub-split of train (customers the base model never saw), so the validation
split evaluates calibration honestly; the method is then selected on
validation Brier and the decision threshold on validation F1 (ADR-0009:
the test split never participates in any of these choices).

The threshold is a provisional operating point: the ADR-0011 decision
policy (issue #11) owns the final cost-based policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from training.metrics import FloatArray

# A fitted scorer is either the bare pipeline ("none") or a calibration
# wrapper around its frozen form; both expose predict_proba.
type FittedScorer = Pipeline | CalibratedClassifierCV

CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")
# Cross-fitting folds of the calibrator over the calibration sub-split.
# sklearn 1.9 note: with a FrozenEstimator there is no "prefit" branch —
# CalibratedClassifierCV still runs cross_val_predict, but the base
# estimator's fit is a no-op, so the calibrator only ever sees x_cal (no
# leakage; verified against sklearn source). The benign side effect is a
# StratifiedKFold UserWarning when a class has fewer members than folds.
CALIBRATION_CV_FOLDS = 5
# Tie-break preference: simpler calibration wins an exact Brier tie.
_COMPLEXITY_ORDER = {"none": 0, "sigmoid": 1, "isotonic": 2}

CALIBRATION_FITTED_ON = (
    "train calibration sub-split (CalibratedClassifierCV over "
    "FrozenEstimator); method selected on validation Brier"
)
THRESHOLD_RULE = "f1_max_on_validation"
THRESHOLD_NOTE = (
    "provisional operating point selected on validation F1; the ADR-0011 "
    "decision policy (issue #11) owns the final cost-based threshold"
)


@dataclass(frozen=True)
class CalibrationChoice:
    method: str
    val_brier_by_method: dict[str, float]
    fitted_on: str = CALIBRATION_FITTED_ON
    cv_folds: int = CALIBRATION_CV_FOLDS

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "val_brier_by_method": self.val_brier_by_method,
            "fitted_on": self.fitted_on,
            "cv_folds": self.cv_folds,
        }


@dataclass(frozen=True)
class ThresholdChoice:
    value: float
    rule: str
    val_f1: float
    note: str = THRESHOLD_NOTE

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "rule": self.rule,
            "val_f1": self.val_f1,
            "note": self.note,
        }


@dataclass(frozen=True)
class AtThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    n_flagged: int

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "n_flagged": self.n_flagged,
        }


def fit_candidate_calibrators(
    pipeline: Pipeline, x_cal: pd.DataFrame, y_cal: FloatArray
) -> dict[str, FittedScorer]:
    """Fit every calibration method on the calibration sub-split.

    ``none`` is the unwrapped pipeline; sigmoid/isotonic wrap its frozen
    form with a calibrator, keeping preprocessing + classifier +
    calibration in a single serializable artifact (ADR-0001).
    """
    fitted: dict[str, FittedScorer] = {"none": pipeline}
    for method in CALIBRATION_METHODS[1:]:
        fitted[method] = CalibratedClassifierCV(
            FrozenEstimator(pipeline), method=method, cv=CALIBRATION_CV_FOLDS
        ).fit(x_cal, y_cal)
    return fitted


def select_calibration(
    val_probs: Mapping[str, FloatArray], y_val: FloatArray
) -> CalibrationChoice:
    """Pick the calibration method with the lowest validation Brier."""
    unknown = sorted(set(val_probs) - set(CALIBRATION_METHODS))
    if unknown:
        raise ValueError(f"unknown calibration methods: {unknown}")
    if not val_probs:
        raise ValueError("at least one calibration method is required")
    brier_by = {
        method: float(brier_score_loss(y_val, probs))
        for method, probs in val_probs.items()
    }
    method = min(
        brier_by,
        key=lambda m: (brier_by[m], _COMPLEXITY_ORDER[m]),
    )
    return CalibrationChoice(method=method, val_brier_by_method=brier_by)


def select_threshold(y_true: FloatArray, y_prob: FloatArray) -> ThresholdChoice:
    """Pick the F1-maximizing threshold on validation probabilities."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # The curve has one extra point (no threshold); score only real ones.
    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(f1))
    return ThresholdChoice(
        value=float(thresholds[best]),
        rule=THRESHOLD_RULE,
        val_f1=float(f1[best]),
    )


def evaluate_at_threshold(
    y_true: FloatArray, y_prob: FloatArray, threshold: float
) -> AtThresholdMetrics:
    """Confusion-style metrics at the frozen operating point."""
    flagged = y_prob >= threshold
    return AtThresholdMetrics(
        threshold=threshold,
        precision=float(precision_score(y_true, flagged, zero_division=0)),
        recall=float(recall_score(y_true, flagged, zero_division=0)),
        f1=float(f1_score(y_true, flagged, zero_division=0)),
        n_flagged=int(flagged.sum()),
    )
