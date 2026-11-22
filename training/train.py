"""Training entry point: CLI + programmatic ``train()``.

Reproducible flow (ADR-0002, ADR-0009, ADR-0013, ADR-0017): canonical dataset
-> client-disjoint stratified splits -> single sklearn Pipeline fitted on the
training split only -> validation metrics with uncertainty -> final, separate
test evaluation (never used for selection) -> versioned artifacts
(``models/<version>/model.joblib`` + ``model_meta.json``).

Determinism is logical: same seed and same dataset reproduce the splits, the
split digests and all metrics exactly. Binary equality of the joblib artifact
across environments is NOT guaranteed (serialization details vary); the
manifest records everything needed to audit logical reproducibility.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

from training.dataset import (
    DatasetProvenance,
    canonical_provenance,
    load_dataset,
    sha256_file,
)
from training.features import (
    FEATURE_SCHEMA_VERSION,
    FEATURES,
    ID_COLUMN,
    TARGET,
    feature_schema_digest,
)
from training.metrics import N_BOOTSTRAP, MetricWithCI, SplitMetrics, evaluate_split
from training.preprocessing import build_pipeline
from training.splits import SplitResult, split_dataset

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

MANIFEST_FILENAME = "model_meta.json"
MODEL_FILENAME = "model.joblib"
MANIFEST_SCHEMA_VERSION = "1"

TARGET_DEFINITION = (
    "Historical churn classification label as distributed in the Telco "
    "Customer Churn dataset: Yes when the customer's record is labeled as "
    "churned, No otherwise."
)
TARGET_LIMITATIONS = (
    "Historical record only. The Telco dataset has no prospective observation "
    "window and no score_at/feature_as_of timestamps: scores produced from it "
    "are probabilities of the historical label, not of churn within any "
    "future horizon (e.g. 'churn in 90 days'). No retention effectiveness can "
    "be derived from this dataset."
)
TEST_NOTE = "final evaluation only; not used for model selection"
DECISION_POLICY_NOTE = (
    "decision threshold and calibration parameters are not defined in this "
    "phase; the decision policy (ADR-0011) lands with issue #11; scores are "
    "probabilities of the historical label"
)
DETERMINISM_NOTE = (
    "logical determinism: same seed+dataset reproduces metrics and split "
    "digests; binary equality of the artifact across environments is not "
    "guaranteed"
)


def _git_info() -> tuple[str, bool | None]:
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
        )
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            != ""
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def _require_columns(df: pd.DataFrame) -> None:
    required = {spec.name for spec in FEATURES} | {ID_COLUMN, TARGET}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"training data is missing contract columns: {missing}")


FloatArray = np.ndarray[tuple[int], np.dtype[np.float64]]


def _encoded_target(df: pd.DataFrame) -> FloatArray:
    encoded = df[TARGET].map({"No": 0.0, "Yes": 1.0})
    if encoded.isna().any():
        raise ValueError(f"{TARGET} must be in {{'Yes', 'No'}}")
    return np.asarray(encoded.to_numpy(dtype=np.float64), dtype=np.float64)


def _positive_proba(pipeline: Pipeline, df: pd.DataFrame) -> FloatArray:
    feature_names = [spec.name for spec in FEATURES]
    proba = np.asarray(pipeline.predict_proba(df[feature_names]), dtype=np.float64)
    return np.asarray(proba[:, 1], dtype=np.float64)


def _split_block(result: SplitResult) -> dict[str, object]:
    return {
        "method": "stratified_by_target_grouped_by_customer_deterministic",
        "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "train": {
            "n": result.train_meta.n,
            "churn_rate": result.train_meta.churn_rate,
            "ids_sha256": result.train_meta.ids_sha256,
        },
        "val": {
            "n": result.val_meta.n,
            "churn_rate": result.val_meta.churn_rate,
            "ids_sha256": result.val_meta.ids_sha256,
        },
        "test": {
            "n": result.test_meta.n,
            "churn_rate": result.test_meta.churn_rate,
            "ids_sha256": result.test_meta.ids_sha256,
        },
    }


def _build_manifest(
    *,
    provenance: DatasetProvenance,
    model: str,
    seed: int,
    version: str,
    trained_at: str,
    git_commit: str,
    git_dirty: bool | None,
    splits: SplitResult,
    val_metrics: SplitMetrics,
    test_metrics: SplitMetrics,
    duration_fit: float,
    duration_total: float,
    artifact_sha256: str,
) -> dict[str, object]:
    test_block = test_metrics.to_dict()
    test_block["note"] = TEST_NOTE
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_version": version,
        "trained_at": trained_at,
        "git": {"commit": git_commit, "dirty": git_dirty},
        "seed": seed,
        "dataset": {
            "name": provenance.name,
            "source_url": provenance.source_url,
            "license": provenance.license,
            "sha256": provenance.sha256,
            "n_rows": provenance.n_rows,
            "unit": provenance.unit,
        },
        "target": {
            "label": TARGET,
            "encoding": {"No": 0, "Yes": 1},
            "definition": TARGET_DEFINITION,
            "limitations": TARGET_LIMITATIONS,
        },
        "splits": _split_block(splits),
        "features": {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_digest(),
            "items": [
                {"name": spec.name, "dtype": spec.dtype, "required": spec.required}
                for spec in FEATURES
            ],
        },
        "preprocessing": {
            "steps": [
                "numeric: SimpleImputer(strategy='median') + StandardScaler()",
                "categorical: SimpleImputer(strategy='most_frequent') + "
                "OneHotEncoder(handle_unknown='ignore')",
                "dtype coercion (incl. TotalCharges blanks -> NaN) applied at "
                "dataset ingestion, outside the Pipeline",
            ],
            "fitted_on": "train only",
        },
        "model": {
            "type": "LogisticRegression" if model == "lr" else model,
            "hyperparameters": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
            "artifact_sha256": artifact_sha256,
            "threshold": None,
            "calibration_parameters": None,
            "decision_policy": DECISION_POLICY_NOTE,
        },
        "metrics": {
            "validation": val_metrics.to_dict(),
            "test_final": test_block,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit-learn": _sklearn_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
            "install": "train with the exact pins of backend/requirements-train.txt",
        },
        "training": {
            "duration_fit_seconds": duration_fit,
            "duration_total_seconds": duration_total,
            "n_boot": N_BOOTSTRAP,
            "determinism": DETERMINISM_NOTE,
        },
    }


def _sklearn_version() -> str:
    import sklearn

    return str(sklearn.__version__)


def train(
    df: pd.DataFrame,
    provenance: DatasetProvenance,
    model: str,
    seed: int,
    version: str,
    output_dir: Path,
) -> dict[str, object]:
    """Train one version end to end and write its artifacts.

    The test split is touched exactly once, after all fitting and validation
    are done; nothing before that step uses it (ADR-0009).
    """
    started = time.perf_counter()
    _require_columns(df)

    splits = split_dataset(df, seed)
    pipeline = build_pipeline(model)

    feature_names = [spec.name for spec in FEATURES]
    fit_started = time.perf_counter()
    pipeline.fit(splits.train[feature_names], _encoded_target(splits.train))
    duration_fit = time.perf_counter() - fit_started

    val_prob = _positive_proba(pipeline, splits.val)
    val_metrics = evaluate_split(_encoded_target(splits.val), val_prob, seed=seed + 1)

    test_prob = _positive_proba(pipeline, splits.test)
    test_metrics = evaluate_split(
        _encoded_target(splits.test), test_prob, seed=seed + 2
    )

    git_commit, git_dirty = _git_info()
    duration_total = time.perf_counter() - started

    version_dir = output_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / MODEL_FILENAME
    joblib.dump(pipeline, model_path)
    artifact_sha256 = sha256_file(model_path)

    manifest = _build_manifest(
        provenance=provenance,
        model=model,
        seed=seed,
        version=version,
        trained_at=datetime.now(UTC).isoformat(),
        git_commit=git_commit,
        git_dirty=git_dirty,
        splits=splits,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        duration_fit=duration_fit,
        duration_total=duration_total,
        artifact_sha256=artifact_sha256,
    )
    (version_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _print_summary(
        model_type="LogisticRegression" if model == "lr" else model,
        version=version,
        seed=seed,
        n_rows=len(df),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        duration_fit=duration_fit,
        duration_total=duration_total,
        version_dir=version_dir,
    )
    return manifest


def _print_summary(
    *,
    model_type: str,
    version: str,
    seed: int,
    n_rows: int,
    val_metrics: SplitMetrics,
    test_metrics: SplitMetrics,
    duration_fit: float,
    duration_total: float,
    version_dir: Path,
) -> None:
    def line(m: MetricWithCI, name: str) -> str:
        return (
            f"  {name}: {m.value:.4f} "
            f"(95% bootstrap CI: {m.ci_low:.4f}-{m.ci_high:.4f})"
        )

    print("=== Training summary (historical evaluation) ===")
    print(f"model: {model_type} (baseline, ADR-0004)")
    print(f"version: {version} | seed: {seed}")
    print(f"dataset rows: {n_rows}")
    print("validation metrics (historical):")
    print(line(val_metrics.roc_auc, "roc_auc"))
    print(line(val_metrics.pr_auc, "pr_auc "))
    print(line(val_metrics.brier, "brier  "))
    print(f"  base_rate: {val_metrics.base_rate:.4f}")
    print("test_final metrics (historical; final evaluation only):")
    print(line(test_metrics.roc_auc, "roc_auc"))
    print(line(test_metrics.pr_auc, "pr_auc "))
    print(line(test_metrics.brier, "brier  "))
    print(f"durations: fit={duration_fit:.2f}s total={duration_total:.2f}s")
    print(
        f"artifacts: {version_dir / MODEL_FILENAME}, "
        f"{version_dir / MANIFEST_FILENAME}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the churn baseline and write versioned artifacts."
    )
    parser.add_argument("--model", choices=["lr"], default="lr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_dataset()
    provenance = canonical_provenance(n_rows=len(df))
    train(
        df=df,
        provenance=provenance,
        model=args.model,
        seed=args.seed,
        version=args.version,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
