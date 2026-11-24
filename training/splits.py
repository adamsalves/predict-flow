"""Deterministic client-disjoint splits, stratified by target (ADR-0009).

The unit of analysis is the customer: every split allocates whole
``customerID`` values, never rows, so no customer appears in more than one
split and no train/validation row is ever counted as independent test
evidence. Allocation is stratified by the ``Churn`` label to preserve the
base rate in each split, and deterministic through ``random.Random`` seeded
per stratum: the same seed and dataset always reproduce the same partition
and the same split digests.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import pandas as pd

from training.features import ID_COLUMN, TARGET

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
# Share of the training split reserved to fit the probability calibrator
# (ADR-0009: exclusive purpose, disjoint customers, recorded in the manifest).
CALIBRATION_RATIO = 0.25


@dataclass(frozen=True)
class SplitMeta:
    n: int
    churn_rate: float
    ids_sha256: str


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_meta: SplitMeta
    val_meta: SplitMeta
    test_meta: SplitMeta


def split_ids_digest(sorted_ids: list[str]) -> str:
    """SHA-256 over the newline-joined customer IDs, for the manifest."""
    return hashlib.sha256("\n".join(sorted_ids).encode("utf-8")).hexdigest()


def _allocate(ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split one stratum into 60/20/20; integer rounding remainder to train."""
    n = len(ids)
    n_val = int(n * VAL_RATIO)
    n_test = int(n * TEST_RATIO)
    n_train = n - n_val - n_test
    return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]


def split_dataset(df: pd.DataFrame, seed: int) -> SplitResult:
    """Partition customers into disjoint train/val/test splits."""
    labels = sorted(df[TARGET].unique().tolist())
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for label in labels:
        stratum_ids = sorted(df.loc[df[TARGET] == label, ID_COLUMN].tolist())
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(stratum_ids)
        train, val, test = _allocate(stratum_ids)
        train_ids.extend(train)
        val_ids.extend(val)
        test_ids.extend(test)

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_set & val_set), "train/val customer overlap"
    assert not (train_set & test_set), "train/test customer overlap"
    assert not (val_set & test_set), "val/test customer overlap"
    assert train_set | val_set | test_set == set(df[ID_COLUMN].tolist()), (
        "splits must cover every customer exactly once"
    )

    def frame(ids: list[str]) -> pd.DataFrame:
        return df[df[ID_COLUMN].isin(ids)]

    def meta(ids: list[str]) -> SplitMeta:
        part = frame(ids)
        return SplitMeta(
            n=len(part),
            churn_rate=float(part[TARGET].eq("Yes").mean()),
            ids_sha256=split_ids_digest(sorted(ids)),
        )

    return SplitResult(
        train=frame(train_ids),
        val=frame(val_ids),
        test=frame(test_ids),
        train_meta=meta(train_ids),
        val_meta=meta(val_ids),
        test_meta=meta(test_ids),
    )


@dataclass(frozen=True)
class CalibrationSplitResult:
    fit: pd.DataFrame
    calibration: pd.DataFrame
    fit_meta: SplitMeta
    calibration_meta: SplitMeta


def split_train_for_calibration(
    train: pd.DataFrame, seed: int, cal_ratio: float = CALIBRATION_RATIO
) -> CalibrationSplitResult:
    """Carve a calibration sub-split out of the training split.

    The calibrator is fitted on customers the base model never saw during
    fitting, so validation evaluates calibration honestly. The sub-split is
    stratified by target, disjoint from the fit rows and deterministic
    through its own seeded stream.
    """
    labels = sorted(train[TARGET].unique().tolist())
    fit_ids: list[str] = []
    cal_ids: list[str] = []
    for label in labels:
        ids = sorted(train.loc[train[TARGET] == label, ID_COLUMN].tolist())
        rng = random.Random(f"{seed}:calibration:{label}")
        rng.shuffle(ids)
        n_cal = int(len(ids) * cal_ratio)
        cal_ids.extend(ids[:n_cal])
        fit_ids.extend(ids[n_cal:])

    fit_set, cal_set = set(fit_ids), set(cal_ids)
    assert not (fit_set & cal_set), "fit/calibration customer overlap"
    assert fit_set | cal_set == set(train[ID_COLUMN].tolist()), (
        "calibration sub-split must cover every training customer exactly once"
    )

    def frame(ids: list[str]) -> pd.DataFrame:
        return train[train[ID_COLUMN].isin(ids)]

    def meta(ids: list[str]) -> SplitMeta:
        part = frame(ids)
        return SplitMeta(
            n=len(part),
            churn_rate=float(part[TARGET].eq("Yes").mean()),
            ids_sha256=split_ids_digest(sorted(ids)),
        )

    return CalibrationSplitResult(
        fit=frame(fit_ids),
        calibration=frame(cal_ids),
        fit_meta=meta(fit_ids),
        calibration_meta=meta(cal_ids),
    )
