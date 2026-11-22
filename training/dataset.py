"""Canonical Telco dataset acquisition, integrity check and contract coercion.

Source: IBM's ``telco-customer-churn-on-icp4d`` code pattern repository
(Apache License 2.0, see the repository LICENSE file). The raw CSV is cached
under ``data/raw/`` (gitignored, ADR-0017): an existing cache whose SHA-256
matches the pinned digest is reused without any network call; any divergence
fails loudly instead of silently re-downloading.

Design decision (recorded): dtype coercion — including
``pd.to_numeric(TotalCharges, errors="coerce")`` for blank cells — happens
here at ingestion, as enforcement of the feature contract. The serialized
sklearn Pipeline (ADR-0004) contains only imputation, encoding and scaling,
so the ``model.joblib`` artifact references sklearn/numpy classes only and
serving stays free of any pandas dependency (ADR-0013).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from training.features import FEATURES, ID_COLUMN, TARGET

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATASET_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DATASET_LICENSE = (
    "Apache License 2.0, as published in the LICENSE file of the IBM "
    "telco-customer-churn-on-icp4d repository"
)
DATASET_NAME = "Telco Customer Churn (IBM)"
DATASET_UNIT = "one historical record per customer"
EXPECTED_ROWS = 7043
CACHE_DIR = Path("data") / "raw"
CACHE_FILENAME = "Telco-Customer-Churn.csv"
DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class DatasetProvenance:
    """Origin and identity of the dataset, recorded in the manifest."""

    name: str
    source_url: str
    license: str
    sha256: str
    n_rows: int
    unit: str


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_download(url: str) -> bytes:
    with urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        content: bytes = response.read()
    return content


def _ensure_cached(
    dest: Path,
    expected_sha256: str,
    download: Callable[[str], bytes],
) -> Path:
    """Return a verified cache file, downloading it only when missing.

    A cached file with a mismatched digest is a hard error: the caller must
    remove it explicitly, so a corrupted or substituted file never flows
    silently into training.
    """
    if dest.is_file():
        digest = sha256_file(dest)
        if digest != expected_sha256:
            raise ValueError(
                f"cached dataset at {dest} has sha256 {digest}, expected "
                f"{expected_sha256}; remove the file to re-download it"
            )
        return dest

    content = download(DATASET_URL)
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"downloaded dataset from {DATASET_URL} has sha256 {digest}, "
            f"expected {expected_sha256}; refusing to cache it"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest


def validate_dataset_contract(df: pd.DataFrame) -> None:
    """Validate the canonical Telco dataset against its documented shape."""
    required_columns = {spec.name for spec in FEATURES} | {ID_COLUMN, TARGET}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if not df[ID_COLUMN].is_unique:
        duplicates = int(df[ID_COLUMN].duplicated().sum())
        raise ValueError(
            f"customerID must be unique (one record per customer); "
            f"found {duplicates} duplicates"
        )
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} rows in the canonical Telco dataset, "
            f"got {len(df)}"
        )
    observed_targets = set(df[TARGET].unique().tolist())
    if not observed_targets <= {"Yes", "No"}:
        raise ValueError(
            f"{TARGET} must be in {{'Yes', 'No'}}, got {sorted(observed_targets)}"
        )


def apply_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to the dtypes declared by the feature contract.

    Blank ``TotalCharges`` cells become NaN here, at the contract boundary;
    the sklearn Pipeline only imputes them afterwards.
    """
    coerced = df.copy()
    for spec in FEATURES:
        if spec.dtype == "int":
            coerced[spec.name] = pd.to_numeric(
                coerced[spec.name], errors="raise"
            ).astype("int64")
        elif spec.dtype == "float":
            # errors="coerce" is deliberate: the source ships blank
            # TotalCharges cells for zero-tenure customers.
            coerced[spec.name] = pd.to_numeric(
                coerced[spec.name], errors="coerce"
            ).astype("float64")
    return coerced


def load_dataset(
    cache_dir: Path = CACHE_DIR,
    download: Callable[[str], bytes] = _default_download,
) -> pd.DataFrame:
    """Load the canonical Telco dataset with contract validation applied."""
    path = _ensure_cached(cache_dir / CACHE_FILENAME, DATASET_SHA256, download)
    df = pd.read_csv(path, dtype="str")
    df = apply_feature_dtypes(df)
    validate_dataset_contract(df)
    return df


def canonical_provenance(n_rows: int) -> DatasetProvenance:
    """Provenance record for the canonical Telco dataset."""
    return DatasetProvenance(
        name=DATASET_NAME,
        source_url=DATASET_URL,
        license=DATASET_LICENSE,
        sha256=DATASET_SHA256,
        n_rows=n_rows,
        unit=DATASET_UNIT,
    )
