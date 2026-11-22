"""Evaluation metrics with uncertainty, on the historical label (ADR-0009).

Every metric carries a 95% percentile bootstrap confidence interval
(``n=1000`` resamples, seeded from a stream derived from the training seed),
plus the base rate and a quantile-binned calibration table. Explicit
tolerances: bootstrap CIs are Monte Carlo estimates — two runs with the
same seed agree exactly, runs with different seeds agree approximately
(monte carlo error only). These metrics describe discrimination and
calibration of the historical label; no prospective horizon is implied.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

FloatArray = NDArray[np.float64]

N_BOOTSTRAP = 1000
CI_PERCENTILE_LOW = 2.5
CI_PERCENTILE_HIGH = 97.5
CALIBRATION_BINS = 10

# Reproducibility tolerance documented for the deterministic-training tests:
# same seed must reproduce CIs exactly; metrics agree under np.allclose with
# default tolerances across seeds.
MetricFn = Callable[[FloatArray, FloatArray], float]


@dataclass(frozen=True)
class MetricWithCI:
    value: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, float]:
        return {"value": self.value, "ci_low": self.ci_low, "ci_high": self.ci_high}


@dataclass(frozen=True)
class CalibrationRow:
    bin: int
    mean_predicted: float
    observed_rate: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "bin": self.bin,
            "mean_predicted": self.mean_predicted,
            "observed_rate": self.observed_rate,
            "n": self.n,
        }


@dataclass(frozen=True)
class SplitMetrics:
    roc_auc: MetricWithCI
    pr_auc: MetricWithCI
    brier: MetricWithCI
    base_rate: float
    calibration: tuple[CalibrationRow, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "roc_auc": self.roc_auc.value,
            "roc_auc_ci95": [self.roc_auc.ci_low, self.roc_auc.ci_high],
            "pr_auc": self.pr_auc.value,
            "pr_auc_ci95": [self.pr_auc.ci_low, self.pr_auc.ci_high],
            "brier": self.brier.value,
            "brier_ci95": [self.brier.ci_low, self.brier.ci_high],
            "base_rate": self.base_rate,
            "calibration": [row.to_dict() for row in self.calibration],
            "methodology": (
                "historical label; roc_auc = roc_auc_score; pr_auc = "
                "average_precision_score; brier = brier_score_loss; "
                "ci95 = percentile bootstrap, n_boot=1000 with seeded rng; "
                "calibration = quantile bins on predicted probability"
            ),
        }


def _metric_with_ci(
    y_true: FloatArray,
    y_prob: FloatArray,
    metric: MetricFn,
    seed: int,
) -> MetricWithCI:
    value = metric(y_true, y_prob)
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    boot: list[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        if np.unique(y_true[idx]).shape[0] < 2:
            continue
        boot.append(metric(y_true[idx], y_prob[idx]))
    ci = np.percentile(
        np.asarray(boot, dtype=np.float64), [CI_PERCENTILE_LOW, CI_PERCENTILE_HIGH]
    )
    return MetricWithCI(value=value, ci_low=float(ci[0]), ci_high=float(ci[1]))


def calibration_table(
    y_true: FloatArray, y_prob: FloatArray, n_bins: int = CALIBRATION_BINS
) -> tuple[CalibrationRow, ...]:
    """Quantile-binned calibration table (predicted mean vs observed rate)."""
    edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.shape[0] < 2:
        return (
            CalibrationRow(
                bin=0,
                mean_predicted=float(y_prob.mean()),
                observed_rate=float(y_true.mean()),
                n=int(y_true.shape[0]),
            ),
        )
    bin_idx = np.digitize(y_prob, edges[1:-1], right=True)
    rows = []
    for b in range(edges.shape[0] - 1):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            CalibrationRow(
                bin=b,
                mean_predicted=float(y_prob[mask].mean()),
                observed_rate=float(y_true[mask].mean()),
                n=count,
            )
        )
    return tuple(rows)


def evaluate_split(
    y_true: FloatArray, y_prob: FloatArray, seed: int
) -> SplitMetrics:
    """Full metric set for one split. ``seed`` derives the bootstrap stream."""
    return SplitMetrics(
        roc_auc=_metric_with_ci(y_true, y_prob, _roc_auc, seed),
        pr_auc=_metric_with_ci(y_true, y_prob, _pr_auc, seed),
        brier=_metric_with_ci(y_true, y_prob, _brier, seed),
        base_rate=float(y_true.mean()),
        calibration=calibration_table(y_true, y_prob),
    )


def _roc_auc(y_true: FloatArray, y_prob: FloatArray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def _pr_auc(y_true: FloatArray, y_prob: FloatArray) -> float:
    return float(average_precision_score(y_true, y_prob))


def _brier(y_true: FloatArray, y_prob: FloatArray) -> float:
    return float(brier_score_loss(y_true, y_prob))
