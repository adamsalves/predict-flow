"""End-to-end training on a synthetic mini dataset (no network).

Determinism tolerances: with the same seed, metrics and split digests must
match EXACTLY (``==``); the metrics comparison also runs under
``np.allclose`` with default tolerances to document that float aggregation
is the only place where equality is not bitwise by design. This is logical
determinism — a reproducible hash of the split IDs is not evidence of the
dataset's provenance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from training.dataset import DatasetProvenance
from training.factors_schema import factors_schema_digest
from training.features import FEATURES, ID_COLUMN, feature_schema_digest
from training.publication import (
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    REGISTRY_FILENAME,
    verify_publication,
)
from training.splits import split_dataset
from training.tests.conftest import FrameFactory
from training.train import train

_SEED = 42
_VERSION = "testrun"


def _provenance() -> DatasetProvenance:
    return DatasetProvenance(
        name="synthetic mini dataset",
        source_url="local://synthetic",
        license="synthetic, no license",
        sha256="0" * 64,
        n_rows=60,
        unit="one synthetic record per customer",
    )


def _run(make_frame: FrameFactory, output_dir: Path) -> dict[str, object]:
    df = make_frame(120, 60)
    return train(
        df=df,
        provenance=_provenance(),
        seed=_SEED,
        version=_VERSION,
        output_dir=output_dir,
    )


def test_end_to_end_writes_artifacts_and_full_manifest(
    make_frame: FrameFactory, tmp_path: Path
) -> None:
    manifest = _run(make_frame, tmp_path / "models")
    version_dir = tmp_path / "models" / _VERSION
    assert (version_dir / MODEL_FILENAME).is_file()
    assert (version_dir / MANIFEST_FILENAME).is_file()

    written = json.loads((version_dir / MANIFEST_FILENAME).read_text())
    assert written == manifest

    # Top-level structure required by ADR-0002 / issues #5 and #6.
    for key in (
        "schema_version",
        "model_version",
        "trained_at",
        "git",
        "seed",
        "dataset",
        "target",
        "splits",
        "features",
        "factors_schema",
        "preprocessing",
        "selection",
        "model",
        "explanation",
        "metrics",
        "environment",
        "training",
    ):
        assert key in written, f"manifest missing key: {key}"

    assert written["schema_version"] == "2"
    assert written["model_version"] == _VERSION
    assert written["seed"] == _SEED
    assert written["git"]["commit"]
    assert isinstance(written["git"]["dirty"], bool)

    dataset = written["dataset"]
    assert dataset["sha256"] == "0" * 64
    assert dataset["n_rows"] == 60
    assert dataset["unit"] == "one synthetic record per customer"

    target = written["target"]
    assert target["label"] == "Churn"
    assert target["encoding"] == {"No": 0, "Yes": 1}
    assert target["definition"]
    limitations = target["limitations"].lower()
    assert "historical" in limitations
    assert "90 days" in limitations  # no prospective horizon is asserted

    # 120 No -> 72/24/24 and 60 Yes -> 36/12/12; calibration carves 18+9.
    splits = written["splits"]
    assert splits["train"]["n"] == 108
    assert splits["val"]["n"] == 36
    assert splits["test"]["n"] == 36
    assert splits["calibration"]["n"] == 27

    features = written["features"]
    assert features["feature_schema_version"] == "1"
    assert features["feature_schema_sha256"] == feature_schema_digest()
    assert len(features["items"]) == 19

    factors = written["factors_schema"]
    assert factors["schema_version"] == "1"
    assert factors["sha256"] == factors_schema_digest()

    assert written["preprocessing"]["fitted_on"] == "train fit sub-split only"

    selection = written["selection"]
    assert selection["winner"] in {
        "lr-baseline",
        "rf-deep",
        "rf-regularized",
    }
    assert selection["gate_rule"]
    assert selection["limitations"]
    assert len(selection["candidates"]) == 3
    assert len(selection["comparisons"]) == 2

    model = written["model"]
    assert model["candidate"] == selection["winner"]
    expected_type = (
        "LogisticRegression"
        if selection["winner"] == "lr-baseline"
        else "RandomForestClassifier"
    )
    assert model["type"] == expected_type
    assert model["hyperparameters"]
    assert 0.0 < model["threshold"]["value"] <= 1.0
    assert model["threshold"]["rule"] == "f1_max_on_validation"
    assert model["calibration"]["method"] in ("none", "sigmoid", "isotonic")
    assert "issue #11" in model["decision_policy"]
    assert "base_rate" not in written["target"]  # lives under metrics only

    explanation = written["explanation"]
    if selection["winner"] == "lr-baseline":
        assert explanation["method"] == "linear"
        assert explanation["unit"] == "log_odds"
    else:
        assert explanation["method"] == "tree_shap"
        assert explanation["unit"] == "probability"

    validation = written["metrics"]["validation"]
    test_final = written["metrics"]["test_final"]
    for block in (validation, test_final):
        for metric in ("roc_auc", "pr_auc", "brier", "base_rate"):
            assert isinstance(block[metric], float)
        assert len(block["roc_auc_ci95"]) == 2
        assert len(block["pr_auc_ci95"]) == 2
        assert len(block["brier_ci95"]) == 2
        assert len(block["calibration"]) >= 1
    assert "not used for model selection" in test_final["note"]
    assert test_final["at_threshold"]["threshold"] == model["threshold"]["value"]

    for package in ("python", "scikit-learn", "numpy", "pandas", "joblib", "shap"):
        assert written["environment"][package]
    assert written["training"]["n_boot"] == 1000
    assert "logical determinism" in written["training"]["determinism"]

    # Publication registry: trusted digest source outside the version dir.
    assert (tmp_path / "models" / REGISTRY_FILENAME).is_file()
    record = verify_publication(version_dir)
    assert record.artifact_sha256 == model["artifact_sha256"]


def test_selection_is_isolated_from_the_test_split(
    make_frame: FrameFactory, tmp_path: Path
) -> None:
    """Corrupting the test split must not change any selection decision:
    winner, calibration and threshold come from train/validation only
    (ADR-0009). The corruption permutes feature values inside the test
    rows only — labels stay intact so the stratified split assignment is
    unchanged — and the test evaluation itself must move, proving the
    corruption landed where intended."""
    df = make_frame(120, 60)
    train(
        df=df,
        provenance=_provenance(),
        seed=_SEED,
        version="clean",
        output_dir=tmp_path / "a",
    )

    test_ids = set(split_dataset(df, seed=_SEED).test[ID_COLUMN])
    corrupted = df.copy()
    mask = corrupted[ID_COLUMN].isin(test_ids)
    rng = np.random.default_rng(0)
    for spec in FEATURES:
        corrupted.loc[mask, spec.name] = rng.permutation(
            corrupted.loc[mask, spec.name].to_numpy()
        )

    train(
        df=corrupted,
        provenance=_provenance(),
        seed=_SEED,
        version="dirty",
        output_dir=tmp_path / "b",
    )

    clean = json.loads(
        (tmp_path / "a" / "clean" / MANIFEST_FILENAME).read_text()
    )
    dirty = json.loads(
        (tmp_path / "b" / "dirty" / MANIFEST_FILENAME).read_text()
    )
    assert clean["selection"] == dirty["selection"]
    assert clean["model"]["threshold"] == dirty["model"]["threshold"]
    assert clean["model"]["calibration"] == dirty["model"]["calibration"]
    clean_auc = clean["metrics"]["test_final"]["roc_auc"]
    dirty_auc = dirty["metrics"]["test_final"]["roc_auc"]
    assert clean_auc != dirty_auc


def test_split_digests_match_recompute(
    make_frame: FrameFactory, tmp_path: Path
) -> None:
    _run(make_frame, tmp_path / "models")
    written = json.loads(
        (tmp_path / "models" / _VERSION / MANIFEST_FILENAME).read_text()
    )
    df = make_frame(120, 60)
    expected = split_dataset(df, seed=_SEED)
    assert written["splits"]["train"]["ids_sha256"] == expected.train_meta.ids_sha256
    assert written["splits"]["val"]["ids_sha256"] == expected.val_meta.ids_sha256
    assert written["splits"]["test"]["ids_sha256"] == expected.test_meta.ids_sha256


def test_manifest_artifact_sha256_matches_written_file(
    make_frame: FrameFactory, tmp_path: Path
) -> None:
    _run(make_frame, tmp_path / "models")
    model_path = tmp_path / "models" / _VERSION / MODEL_FILENAME
    written = json.loads(
        (tmp_path / "models" / _VERSION / MANIFEST_FILENAME).read_text()
    )
    expected = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert written["model"]["artifact_sha256"] == expected


def test_model_joblib_loads_outside_repo_import_path(
    make_frame: FrameFactory, tmp_path: Path
) -> None:
    """The artifact must load and predict without the ``training`` package
    on sys.path (ADR-0001/ADR-0017): the pickle references only
    sklearn/numpy classes, never training code or lambdas."""
    df = make_frame(120, 60)
    train(
        df=df,
        provenance=_provenance(),
        seed=_SEED,
        version=_VERSION,
        output_dir=tmp_path / "models",
    )
    model_path = tmp_path / "models" / _VERSION / MODEL_FILENAME
    feature_names = [spec.name for spec in FEATURES]
    input_csv = tmp_path / "scoring_input.csv"
    df.head(3)[feature_names].to_csv(input_csv, index=False)

    script = (
        "import joblib\n"
        "import pandas as pd\n"
        f"df = pd.read_csv({str(input_csv)!r})\n"
        f"pipe = joblib.load({str(model_path)!r})\n"
        "proba = pipe.predict_proba(df)\n"
        "assert proba.shape[0] == 3\n"
        "print('PORTABILITY_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PORTABILITY_OK" in result.stdout


def test_logical_determinism_same_seed(
    make_frame: FrameFactory, tmp_path: Path
) -> None:
    _run(make_frame, tmp_path / "a")
    _run(make_frame, tmp_path / "b")

    stable_a = json.loads(
        (tmp_path / "a" / _VERSION / MANIFEST_FILENAME).read_text()
    )
    stable_b = json.loads(
        (tmp_path / "b" / _VERSION / MANIFEST_FILENAME).read_text()
    )
    for block in (stable_a, stable_b):
        block.pop("trained_at")
        block["training"].pop("duration_fit_seconds_by_candidate")
        block["training"].pop("duration_total_seconds")
        block["git"] = {"commit": "pinned", "dirty": False}
        # Binary artifact equality is explicitly not part of logical
        # determinism (see the manifest determinism note); the hash stays a
        # string and is verified against the on-disk file elsewhere.
        block["model"].pop("artifact_sha256")
    assert stable_a == stable_b

    # Explicit metric comparison under documented tolerances.
    for split in ("validation", "test_final"):
        for metric in ("roc_auc", "pr_auc", "brier"):
            assert np.allclose(
                stable_a["metrics"][split][metric],
                stable_b["metrics"][split][metric],
                rtol=1e-5,
                atol=1e-8,
            )
    assert (
        stable_a["splits"]["train"]["ids_sha256"]
        == stable_b["splits"]["train"]["ids_sha256"]
    )
