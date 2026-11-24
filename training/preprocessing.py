"""Single sklearn Pipeline: preprocessing + classifier, fitted on train only
(ADR-0004). The serialized artifact contains the whole preprocessing chain,
so serving runs exactly the same imputation, encoding and scaling learned
during training — nothing is re-implemented on the backend side.

Feature routing note: the contract (``training/features.py``) declares
``SeniorCitizen`` as int; as a modeling decision it is routed through the
categorical branch (low-cardinality discrete variable) while the continuous
charges/tenure features are scaled. Dtype coercion itself stays outside the
Pipeline, at dataset ingestion (see ``training/dataset.py``).
"""

from __future__ import annotations

from collections.abc import Mapping

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Continuous features: median imputation + standard scaling.
PIPELINE_NUMERIC_FEATURES: tuple[str, ...] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)
# All remaining contract features go through the categorical branch
# (mode imputation + one-hot with unknown-category tolerance at serve time).
PIPELINE_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)


def build_preprocessor() -> ColumnTransformer:
    """Build the shared preprocessing block used by every candidate."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(PIPELINE_NUMERIC_FEATURES),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(PIPELINE_CATEGORICAL_FEATURES),
            ),
        ]
    )


def build_pipeline(
    model: str = "lr",
    hyperparameters: Mapping[str, object] | None = None,
    *,
    seed: int = 42,
) -> Pipeline:
    """Build the single sklearn Pipeline for the chosen model.

    ``lr`` is the mandatory baseline and ``rf`` the challenger (ADR-0004);
    both share the exact same preprocessing block so the validation gate
    compares classifiers, not feature pipelines. ``random_state`` is always
    derived from ``seed`` for the forest; explicit hyperparameters override
    the defaults.
    """
    params = dict(hyperparameters or {})
    if model == "lr":
        classifier = LogisticRegression(
            **{"C": 1.0, "max_iter": 1000, "solver": "lbfgs", **params}
        )
    elif model == "rf":
        classifier = RandomForestClassifier(
            **{"n_estimators": 200, "random_state": seed, **params}
        )
    else:
        raise ValueError(f"unsupported model {model!r}; choices: 'lr', 'rf'")
    return Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)]
    )
