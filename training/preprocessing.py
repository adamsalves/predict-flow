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

from sklearn.compose import ColumnTransformer
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


def build_pipeline(model: str = "lr") -> Pipeline:
    """Build the single sklearn Pipeline for the chosen model.

    Only ``lr`` exists today (the mandatory baseline, ADR-0004); issue #6
    adds ``rf`` as a challenger behind the same preprocessing.
    """
    if model != "lr":
        raise ValueError(f"unsupported model {model!r}; choices: 'lr'")

    preprocessor = ColumnTransformer(
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
    classifier = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
