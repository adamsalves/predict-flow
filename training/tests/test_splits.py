"""Split disjointness, determinism and ratio tests (ADR-0009)."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from training.features import ID_COLUMN, TARGET
from training.splits import (
    split_dataset,
    split_ids_digest,
    split_train_for_calibration,
)
from training.tests.conftest import FrameFactory


def _ids(df: pd.DataFrame) -> set[str]:
    return set(df[ID_COLUMN].tolist())


def test_splits_are_totally_disjoint_and_cover_all_customers(
    make_frame: FrameFactory,
) -> None:
    df = make_frame(600, 150)
    result = split_dataset(df, seed=42)

    train_ids = _ids(result.train)
    val_ids = _ids(result.val)
    test_ids = _ids(result.test)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == _ids(df)
    # No customer counted twice overall.
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(df)


def test_determinism_same_seed_same_digests(make_frame: FrameFactory) -> None:
    df = make_frame(600, 150)
    first = split_dataset(df, seed=42)
    second = split_dataset(df, seed=42)
    assert first.train_meta.ids_sha256 == second.train_meta.ids_sha256
    assert first.val_meta.ids_sha256 == second.val_meta.ids_sha256
    assert first.test_meta.ids_sha256 == second.test_meta.ids_sha256
    assert sorted(_ids(first.train)) == sorted(_ids(second.train))


def test_different_seed_changes_partition(make_frame: FrameFactory) -> None:
    df = make_frame(600, 150)
    baseline = split_dataset(df, seed=42)
    other = split_dataset(df, seed=7)
    assert baseline.train_meta.ids_sha256 != other.train_meta.ids_sha256


def test_stratified_ratios_with_remainder_to_train(
    make_frame: FrameFactory,
) -> None:
    df = make_frame(600, 150)
    result = split_dataset(df, seed=42)
    # 600 No -> 360/120/120 and 150 Yes -> 90/30/30 exactly.
    assert len(result.train) == 450
    assert len(result.val) == 150
    assert len(result.test) == 150
    # Stratification preserves the base rate in every split.
    assert result.train_meta.churn_rate == pytest.approx(0.2)
    assert result.val_meta.churn_rate == pytest.approx(0.2)
    assert result.test_meta.churn_rate == pytest.approx(0.2)


def test_imbalanced_small_strata_stay_disjoint(make_frame: FrameFactory) -> None:
    df = make_frame(9, 5)
    result = split_dataset(df, seed=42)
    total = len(result.train) + len(result.val) + len(result.test)
    assert total == len(df)
    assert _ids(result.train).isdisjoint(_ids(result.test))
    # Remainder goes to train: No=9 -> 7/1/1, Yes=5 -> 3/1/1.
    assert len(result.train) == 10
    assert len(result.val) == 2
    assert len(result.test) == 2


def test_split_ids_digest_matches_manual_recompute() -> None:
    ids = ["c", "a", "b"]
    expected = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    assert split_ids_digest(sorted(ids)) == expected


def test_split_ids_digest_changes_with_content() -> None:
    assert split_ids_digest(["a", "b", "c"]) != split_ids_digest(["a", "b", "d"])


def test_target_column_survives_splitting(make_frame: FrameFactory) -> None:
    df = make_frame(600, 150)
    result = split_dataset(df, seed=42)
    assert TARGET in result.train.columns
    assert result.train[TARGET].isin(["Yes", "No"]).all()


def test_calibration_sub_split_is_disjoint_and_covers_train(
    make_frame: FrameFactory,
) -> None:
    df = make_frame(600, 150)
    train = split_dataset(df, seed=42).train
    result = split_train_for_calibration(train, seed=42)

    fit_ids = _ids(result.fit)
    cal_ids = _ids(result.calibration)
    assert fit_ids.isdisjoint(cal_ids)
    assert fit_ids | cal_ids == _ids(train)


def test_calibration_sub_split_is_deterministic_and_stratified(
    make_frame: FrameFactory,
) -> None:
    df = make_frame(600, 150)
    train = split_dataset(df, seed=42).train
    first = split_train_for_calibration(train, seed=42)
    second = split_train_for_calibration(train, seed=42)
    assert first.fit_meta.ids_sha256 == second.fit_meta.ids_sha256
    assert first.calibration_meta.ids_sha256 == second.calibration_meta.ids_sha256
    # Train has 360 No + 90 Yes: int(360*0.25)=90 and int(90*0.25)=22.
    assert len(first.calibration) == 90 + 22
    assert first.calibration_meta.churn_rate == pytest.approx(22 / 112)
    other = split_train_for_calibration(train, seed=7)
    assert other.calibration_meta.ids_sha256 != first.calibration_meta.ids_sha256
