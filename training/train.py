"""Training entry point: CLI + programmatic ``train()``.

Reproducible flow (ADR-0002, ADR-0004, ADR-0009, ADR-0013, ADR-0017):
canonical dataset -> client-disjoint stratified splits -> a calibration
sub-split carved from train -> every candidate (LR baseline + RF
challengers) fitted on the same preprocessing Pipeline -> calibration
method selected on validation Brier -> validation gate on PR-AUC with
bootstrap uncertainty and a calibration veto -> decision threshold selected
on validation F1 -> the frozen winner evaluated exactly once on the test
split -> versioned artifacts (``models/<version>/model.joblib`` +
``model_meta.json``) -> digests published to the trusted registry outside
the version directory.

Determinism is logical: same seed and same dataset reproduce the splits,
the split digests and all metrics exactly. Binary equality of the joblib
artifact across environments is NOT guaranteed (serialization details
vary); the manifest records everything needed to audit logical
reproducibility.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from training.calibration import (
    CalibrationChoice,
    FittedScorer,
    evaluate_at_threshold,
    fit_candidate_calibrators,
    select_calibration,
    select_threshold,
)
from training.candidates import CandidateSpec, default_candidates
from training.dataset import (
    DatasetProvenance,
    canonical_provenance,
    load_dataset,
    sha256_file,
)
from training.factors_schema import (
    EXPLANATION_BY_MODEL,
    factors_schema_digest,
    factors_schema_version,
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
from training.publication import (
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    publish,
)
from training.selection import CandidateEvaluation, GateDecision, decide
from training.splits import (
    CalibrationSplitResult,
    SplitResult,
    split_dataset,
    split_train_for_calibration,
)

MANIFEST_SCHEMA_VERSION = "2"
DEFAULT_TOP_K = 10
# Distinct bootstrap streams: validation uses seed+1+index per candidate,
# the single final test evaluation uses a fixed far offset.
_TEST_METRICS_SEED_OFFSET = 101

MODEL_TYPE_NAMES = {"lr": "LogisticRegression", "rf": "RandomForestClassifier"}

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
    "the operating threshold is the provisional validation F1-max point and "
    "the calibration method was selected on validation Brier; the cost-based "
    "decision policy (ADR-0011) lands with issue #11; scores are "
    "probabilities of the historical label"
)
EXPLANATION_NOTE = (
    "local factors explain the base estimator output through the stable DTO "
    "(ADR-0018); they are predictive, never causal (ADR-0011)"
)
DETERMINISM_NOTE = (
    "logical determinism: same seed+dataset reproduces metrics and split "
    "digests; binary equality of the artifact across environments is not "
    "guaranteed"
)

FloatArray = np.ndarray[tuple[int], np.dtype[np.float64]]


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


def _encoded_target(df: pd.DataFrame) -> FloatArray:
    encoded = df[TARGET].map({"No": 0.0, "Yes": 1.0})
    if encoded.isna().any():
        raise ValueError(f"{TARGET} must be in {{'Yes', 'No'}}")
    return np.asarray(encoded.to_numpy(dtype=np.float64), dtype=np.float64)


def _positive_proba(scorer: FittedScorer, df: pd.DataFrame) -> FloatArray:
    feature_names = [spec.name for spec in FEATURES]
    proba = np.asarray(scorer.predict_proba(df[feature_names]), dtype=np.float64)
    return np.asarray(proba[:, 1], dtype=np.float64)


def _split_block(
    result: SplitResult, calibration: CalibrationSplitResult
) -> dict[str, object]:
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
        "calibration": {
            "n": calibration.calibration_meta.n,
            "churn_rate": calibration.calibration_meta.churn_rate,
            "ids_sha256": calibration.calibration_meta.ids_sha256,
            "note": (
                "carved from the training split to fit the probability "
                "calibrator; disjoint from the fit rows"
            ),
        },
    }


def _build_manifest(
    *,
    provenance: DatasetProvenance,
    seed: int,
    version: str,
    trained_at: str,
    git_commit: str,
    git_dirty: bool | None,
    splits: SplitResult,
    calibration_split: CalibrationSplitResult,
    decision: GateDecision,
    evaluations: tuple[CandidateEvaluation, ...],
    winner_calibration: CalibrationChoice,
    threshold: dict[str, object],
    val_metrics: SplitMetrics,
    test_metrics: SplitMetrics,
    at_threshold: dict[str, object],
    duration_fit_by_candidate: dict[str, float],
    duration_total: float,
    artifact_sha256: str,
) -> dict[str, object]:
    winner = next(e for e in evaluations if e.name == decision.winner)
    test_block = test_metrics.to_dict()
    test_block["note"] = TEST_NOTE
    test_block["at_threshold"] = at_threshold
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
        "splits": _split_block(splits, calibration_split),
        "features": {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": feature_schema_digest(),
            "items": [
                {"name": spec.name, "dtype": spec.dtype, "required": spec.required}
                for spec in FEATURES
            ],
        },
        "factors_schema": {
            "schema_version": factors_schema_version(),
            "sha256": factors_schema_digest(),
        },
        "preprocessing": {
            "steps": [
                "numeric: SimpleImputer(strategy='median') + StandardScaler()",
                "categorical: SimpleImputer(strategy='most_frequent') + "
                "OneHotEncoder(handle_unknown='ignore')",
                "dtype coercion (incl. TotalCharges blanks -> NaN) applied at "
                "dataset ingestion, outside the Pipeline",
            ],
            "fitted_on": "train fit sub-split only",
        },
        "selection": decision.to_dict(evaluations),
        "model": {
            "type": MODEL_TYPE_NAMES[winner.model],
            "candidate": decision.winner,
            "hyperparameters": winner.hyperparameters,
            "artifact_sha256": artifact_sha256,
            "threshold": threshold,
            "calibration": winner_calibration.to_dict(),
            "decision_policy": DECISION_POLICY_NOTE,
        },
        "explanation": {
            "adapter": "app.services.explain:explain_local",
            "method": EXPLANATION_BY_MODEL[winner.model]["method"],
            "unit": EXPLANATION_BY_MODEL[winner.model]["unit"],
            "top_k_default": DEFAULT_TOP_K,
            "note": EXPLANATION_NOTE,
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
            "shap": importlib.metadata.version("shap"),
            "install": "train with the exact pins of backend/requirements-train.txt",
        },
        "training": {
            "duration_fit_seconds_by_candidate": duration_fit_by_candidate,
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
    seed: int,
    version: str,
    output_dir: Path,
    *,
    candidates: tuple[CandidateSpec, ...] | None = None,
) -> dict[str, object]:
    """Train one version end to end and write its artifacts.

    The test split is touched exactly once, after the winner, its
    calibration and its threshold are frozen; nothing before that step
    uses it (ADR-0009).
    """
    started = time.perf_counter()
    _require_columns(df)
    candidate_specs = candidates if candidates is not None else default_candidates()
    feature_names = [spec.name for spec in FEATURES]

    splits = split_dataset(df, seed)
    calibration_split = split_train_for_calibration(splits.train, seed)
    y_fit = _encoded_target(calibration_split.fit)
    y_cal = _encoded_target(calibration_split.calibration)
    y_val = _encoded_target(splits.val)
    y_test = _encoded_target(splits.test)

    evaluations: list[CandidateEvaluation] = []
    scorers: dict[str, FittedScorer] = {}
    calibration_choices: dict[str, CalibrationChoice] = {}
    val_probs_by_candidate: dict[str, FloatArray] = {}
    duration_fit_by_candidate: dict[str, float] = {}

    for index, candidate in enumerate(candidate_specs):
        pipeline = build_pipeline_for(candidate, seed)
        fit_started = time.perf_counter()
        pipeline.fit(calibration_split.fit[feature_names], y_fit)
        duration_fit_by_candidate[candidate.name] = (
            time.perf_counter() - fit_started
        )

        calibrators = fit_candidate_calibrators(
            pipeline, calibration_split.calibration[feature_names], y_cal
        )
        val_probs = {
            method: _positive_proba(scorer, splits.val)
            for method, scorer in calibrators.items()
        }
        choice = select_calibration(val_probs, y_val)
        scorer = calibrators[choice.method]
        val_metrics = evaluate_split(
            y_val, val_probs[choice.method], seed=seed + 1 + index
        )

        evaluations.append(
            CandidateEvaluation(
                name=candidate.name,
                model=candidate.model,
                hyperparameters=candidate.hyperparameters,
                val_metrics=val_metrics,
            )
        )
        scorers[candidate.name] = scorer
        calibration_choices[candidate.name] = choice
        val_probs_by_candidate[candidate.name] = val_probs[choice.method]

    evaluation_tuple = tuple(evaluations)
    decision = decide(evaluation_tuple)
    winner_scorer = scorers[decision.winner]
    winner_calibration = calibration_choices[decision.winner]
    threshold = select_threshold(y_val, val_probs_by_candidate[decision.winner])

    # The frozen winner meets the test split exactly once, here.
    test_prob = _positive_proba(winner_scorer, splits.test)
    test_metrics = evaluate_split(
        y_test, test_prob, seed=seed + _TEST_METRICS_SEED_OFFSET
    )
    at_threshold = evaluate_at_threshold(y_test, test_prob, threshold.value)

    git_commit, git_dirty = _git_info()
    duration_total = time.perf_counter() - started

    version_dir = output_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / MODEL_FILENAME
    joblib.dump(winner_scorer, model_path)
    artifact_sha256 = sha256_file(model_path)

    manifest = _build_manifest(
        provenance=provenance,
        seed=seed,
        version=version,
        trained_at=datetime.now(UTC).isoformat(),
        git_commit=git_commit,
        git_dirty=git_dirty,
        splits=splits,
        calibration_split=calibration_split,
        decision=decision,
        evaluations=evaluation_tuple,
        winner_calibration=winner_calibration,
        threshold=threshold.to_dict(),
        val_metrics=next(
            e.val_metrics for e in evaluation_tuple if e.name == decision.winner
        ),
        test_metrics=test_metrics,
        at_threshold=at_threshold.to_dict(),
        duration_fit_by_candidate=duration_fit_by_candidate,
        duration_total=duration_total,
        artifact_sha256=artifact_sha256,
    )
    (version_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    record = publish(version_dir, git_commit=git_commit)
    _print_summary(
        version=version,
        seed=seed,
        n_rows=len(df),
        evaluations=evaluation_tuple,
        decision=decision,
        calibration=winner_calibration,
        threshold_value=threshold.value,
        test_metrics=test_metrics,
        duration_total=duration_total,
        version_dir=version_dir,
        registry_digest=record.artifact_sha256,
    )
    return manifest


def build_pipeline_for(candidate: CandidateSpec, seed: int) -> FittedScorer:
    return build_pipeline(candidate.model, candidate.hyperparameters, seed=seed)


def _print_summary(
    *,
    version: str,
    seed: int,
    n_rows: int,
    evaluations: tuple[CandidateEvaluation, ...],
    decision: GateDecision,
    calibration: CalibrationChoice,
    threshold_value: float,
    test_metrics: SplitMetrics,
    duration_total: float,
    version_dir: Path,
    registry_digest: str,
) -> None:
    def line(m: MetricWithCI, name: str) -> str:
        return (
            f"  {name}: {m.value:.4f} "
            f"(95% bootstrap CI: {m.ci_low:.4f}-{m.ci_high:.4f})"
        )

    print("=== Training summary (historical evaluation) ===")
    print(f"version: {version} | seed: {seed} | dataset rows: {n_rows}")
    print("validation gate (PR-AUC, calibrated):")
    for evaluation in evaluations:
        marker = "*" if evaluation.name == decision.winner else " "
        pr_auc = evaluation.val_metrics.pr_auc
        print(
            f" {marker} {evaluation.name}: pr_auc={pr_auc.value:.4f} "
            f"[{pr_auc.ci_low:.4f}-{pr_auc.ci_high:.4f}] "
            f"brier={evaluation.val_metrics.brier.value:.4f}"
        )
    outcome = "promoted" if decision.promoted else "baseline kept"
    print(f"decision: {decision.winner} ({outcome}) — {decision.reason}")
    print(f"calibration: {calibration.method} | threshold: {threshold_value:.4f}")
    print("test_final metrics (historical; final evaluation only):")
    print(line(test_metrics.roc_auc, "roc_auc"))
    print(line(test_metrics.pr_auc, "pr_auc "))
    print(line(test_metrics.brier, "brier  "))
    print(f"durations: total={duration_total:.2f}s")
    print(
        f"artifacts: {version_dir / MODEL_FILENAME}, "
        f"{version_dir / MANIFEST_FILENAME}"
    )
    print(f"published digest (trusted registry): {registry_digest}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the churn candidates, run the validation gate and write "
            "the versioned release artifacts."
        )
    )
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
        seed=args.seed,
        version=args.version,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
