"""Local explanation adapters behind the stable DTO (ADR-0018, ADR-0001).

The adapters never reimplement transformations: they invoke the fitted
components inside the single sklearn Pipeline (preprocessing branches and
classifier) on the raw input, then group the encoded-column contributions
back onto contract features. LogisticRegression gets exact linear
contributions (``coef * value``) in log-odds; RandomForest gets TreeSHAP
values in probability units. Everything is derived by introspecting the
fitted artifact, so this module stays free of training-side constants and
of pandas (the serving environment has neither).
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Mapping

import numpy as np
import shap
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.schemas.explanation import (
    CALIBRATED_NOTE,
    METHOD_LINEAR,
    METHOD_TREE_SHAP,
    PREDICTIVE_NOT_CAUSAL_NOTE,
    UNIT_LOG_ODDS,
    UNIT_PROBABILITY,
    ExplanationFactor,
    LocalExplanation,
)

FloatArray = NDArray[np.float64]
RawFeatures = Mapping[str, object]
DEFAULT_TOP_K = 10


def _unwrap_pipeline(scorer: object) -> Pipeline:
    """Return the base Pipeline inside a (possibly calibrated) scorer."""
    obj = scorer
    while not isinstance(obj, Pipeline):
        inner = getattr(obj, "estimator", None)
        if inner is None:
            raise ValueError("scorer does not wrap a sklearn Pipeline")
        obj = inner
    return obj


def _branch_kind(branch: Pipeline) -> str:
    for _, step in branch.steps:
        if isinstance(step, OneHotEncoder):
            return "categorical"
        if isinstance(step, StandardScaler):
            return "numeric"
    raise ValueError("unsupported preprocessing branch: no known transformer")


def _transform_row(preprocessor: ColumnTransformer, raw: RawFeatures) -> FloatArray:
    """Apply the fitted preprocessing branches to one raw input row."""
    outputs: list[FloatArray] = []
    for _, branch, columns in preprocessor.transformers_:
        kind = _branch_kind(branch)
        row: list[object] = []
        for column in columns:
            value = raw.get(column)
            if value is None:
                row.append(np.nan)
            elif kind == "numeric":
                if not isinstance(value, (str, numbers.Real)):
                    raise ValueError(f"non-numeric value for {column}: {value!r}")
                row.append(float(value))
            else:
                # Categorical values pass through unchanged: the fitted
                # encoder matches the native contract types (e.g. int for
                # SeniorCitizen), so coercing to str would miss categories.
                row.append(value)
        x = np.asarray([row], dtype=np.float64 if kind == "numeric" else object)
        with warnings.catch_warnings():
            # Intentional: the fitted branches accept nameless arrays; the
            # column order comes from the fitted transformer itself.
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            transformed = branch.transform(x)
        dense = (
            transformed.toarray() if hasattr(transformed, "toarray") else transformed
        )
        outputs.append(np.asarray(dense, dtype=np.float64))
    return np.hstack(outputs)


def _source_feature(kind: str, columns: list[str], encoded_name: str) -> str:
    if kind == "numeric":
        # Imputation/scaling preserve the source column names.
        return encoded_name
    matches = [c for c in columns if encoded_name.startswith(f"{c}_")]
    if not matches:
        raise ValueError(
            f"cannot map encoded column {encoded_name!r} to a source feature"
        )
    return max(matches, key=len)


def _encoded_groups(preprocessor: ColumnTransformer) -> list[tuple[str, list[int]]]:
    """Group encoded column indices by their contract feature."""
    groups: dict[str, list[int]] = {}
    offset = 0
    for _, branch, columns in preprocessor.transformers_:
        kind = _branch_kind(branch)
        names = list(branch.get_feature_names_out())
        for i, encoded_name in enumerate(names):
            source = _source_feature(kind, list(columns), str(encoded_name))
            groups.setdefault(source, []).append(offset + i)
        offset += len(names)
    return sorted(groups.items())


def _linear_contributions(
    classifier: LogisticRegression, x: FloatArray
) -> tuple[float, FloatArray]:
    coef = np.asarray(classifier.coef_[0], dtype=np.float64)
    base = float(classifier.intercept_[0])
    return base, np.asarray(coef * x[0], dtype=np.float64)


def _tree_shap_contributions(
    classifier: RandomForestClassifier, x: FloatArray
) -> tuple[float, FloatArray]:
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        positive = np.asarray(values[1], dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
        positive = array[..., 1] if array.ndim == 3 else array
    expected = np.atleast_1d(np.asarray(explainer.expected_value, dtype=np.float64))
    base = float(expected[1] if expected.shape[0] > 1 else expected[0])
    return base, np.asarray(positive[0], dtype=np.float64)


def _display_value(value: object) -> str | int | float | None:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, (str, int, float)):
        return value
    return str(value)


def explain_local(
    scorer: object,
    raw: RawFeatures,
    *,
    model_version: str,
    top_k: int = DEFAULT_TOP_K,
) -> LocalExplanation:
    """Explain one score with the adapter matching the active classifier.

    ``scorer`` is the exported artifact: the bare Pipeline or a calibrated
    wrapper around it. Contributions always describe the base estimator's
    output (calibration is a monotone remap, noted in the payload).
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    pipeline = _unwrap_pipeline(scorer)
    calibrated = pipeline is not scorer
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    x = _transform_row(preprocessor, raw)

    if isinstance(classifier, LogisticRegression):
        method, unit = METHOD_LINEAR, UNIT_LOG_ODDS
        base, contributions = _linear_contributions(classifier, x)
    elif isinstance(classifier, RandomForestClassifier):
        method, unit = METHOD_TREE_SHAP, UNIT_PROBABILITY
        base, contributions = _tree_shap_contributions(classifier, x)
    else:
        raise ValueError(
            "no explanation adapter for classifier "
            f"{type(classifier).__name__}"
        )

    factors = [
        ExplanationFactor(
            feature=source,
            value=_display_value(raw.get(source)),
            contribution=float(contributions[indices].sum()),
        )
        for source, indices in _encoded_groups(preprocessor)
    ]
    # Deterministic order: strongest factors first, name as tie-break.
    factors.sort(key=lambda f: (-abs(f.contribution), f.feature))
    note = PREDICTIVE_NOT_CAUSAL_NOTE + (CALIBRATED_NOTE if calibrated else "")
    return LocalExplanation(
        method=method,
        unit=unit,
        model_version=model_version,
        base_value=base,
        output_value=base + float(contributions.sum()),
        top_k=top_k,
        factors=factors[:top_k],
        interpretation_note=note,
    )
