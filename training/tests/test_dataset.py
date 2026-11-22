"""Dataset acquisition, caching and contract validation tests."""

from __future__ import annotations

import hashlib
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from training.dataset import (
    DATASET_SHA256,
    EXPECTED_ROWS,
    _ensure_cached,
    apply_feature_dtypes,
    load_dataset,
    sha256_file,
    validate_dataset_contract,
)
from training.features import ID_COLUMN, TARGET


def _synthetic_csv(n_no: int = 5000, n_yes: int = 2043) -> str:
    n = n_no + n_yes
    header = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,"
        "PhoneService,MultipleLines,InternetService,OnlineSecurity,"
        "OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges,Churn"
    )
    rows = []
    for i in range(n):
        churn = "No" if i < n_no else "Yes"
        total = '""' if i % 13 == 0 else f"{float(i * 11):.2f}"
        rows.append(
            f"CUST-{i:05d},Female,0,Yes,No,{i % 72},Yes,No,DSL,No,Yes,No,Yes,"
            f"No,No,Month-to-month,Yes,Electronic check,{float(20 + i):.2f},"
            f"{total},{churn}"
        )
    return "\n".join([header, *rows]) + "\n"


def test_cached_file_with_mismatched_hash_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_text = _synthetic_csv()
    monkeypatch.setattr(
        "training.dataset.DATASET_SHA256",
        hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
    )
    cache = tmp_path / "Telco-Customer-Churn.csv"
    cache.write_text("not the dataset\n", encoding="utf-8")

    def forbidden_download(url: str) -> bytes:
        raise AssertionError("must not download when a cache file exists")

    with pytest.raises(ValueError, match="cached dataset"):
        _ensure_cached(cache, DATASET_SHA256, forbidden_download)


def test_downloads_once_then_hits_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_text = _synthetic_csv()
    monkeypatch.setattr(
        "training.dataset.DATASET_SHA256",
        hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
    )
    calls = 0

    def counting_download(url: str) -> bytes:
        nonlocal calls
        calls += 1
        return csv_text.encode("utf-8")

    df_first = load_dataset(cache_dir=tmp_path, download=counting_download)
    assert calls == 1
    assert len(df_first) == EXPECTED_ROWS

    df_second = load_dataset(cache_dir=tmp_path, download=counting_download)
    assert calls == 1, "cache hit must not re-download"
    pd.testing.assert_frame_equal(df_first, df_second)


def test_rejects_substituted_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_text = _synthetic_csv()
    monkeypatch.setattr(
        "training.dataset.DATASET_SHA256",
        hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
    )
    with pytest.raises(ValueError, match="refusing to cache"):
        load_dataset(
            cache_dir=tmp_path,
            download=lambda url: b"substituted content",
        )


def test_contract_validation_rejects_duplicate_ids() -> None:
    df = _valid_frame()
    df.at[1, ID_COLUMN] = str(df.at[0, ID_COLUMN])
    with pytest.raises(ValueError, match="unique"):
        validate_dataset_contract(df)


def test_contract_validation_rejects_missing_column() -> None:
    df = _valid_frame().drop(columns=["tenure"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset_contract(df)


def test_contract_validation_rejects_bad_target() -> None:
    df = _valid_frame()
    df.loc[0, TARGET] = "Maybe"
    with pytest.raises(ValueError, match="Yes.*No"):
        validate_dataset_contract(df)


def test_contract_validation_rejects_wrong_row_count() -> None:
    df = _valid_frame().iloc[:-1]
    with pytest.raises(ValueError, match="expected"):
        validate_dataset_contract(df)


def test_contract_validation_accepts_synthetic_frame() -> None:
    validate_dataset_contract(_valid_frame())


def test_dtype_coercion_blanks_become_nan() -> None:
    coerced = apply_feature_dtypes(_valid_frame())
    assert coerced["tenure"].dtype == "int64"
    assert coerced["TotalCharges"].dtype == "float64"
    assert coerced["TotalCharges"].isna().sum() == sum(
        1 for i in range(EXPECTED_ROWS) if i % 13 == 0
    )


def test_real_telco_dataset_downloads_and_validates(tmp_path: Path) -> None:
    """Live check against the canonical source; skipped when offline."""
    try:
        df = load_dataset(cache_dir=tmp_path)
    except OSError as exc:
        pytest.skip(f"network unavailable: {exc}")
    assert len(df) == EXPECTED_ROWS
    assert sha256_file(tmp_path / "Telco-Customer-Churn.csv") == DATASET_SHA256
    assert df["TotalCharges"].isna().sum() > 0


def _valid_frame() -> pd.DataFrame:
    return pd.read_csv(StringIO(_synthetic_csv()), dtype="str")
