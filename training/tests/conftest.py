"""Shared fixtures for training tests: small synthetic Telco-shaped frames."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from training.features import FEATURES, ID_COLUMN, TARGET

FrameFactory = Callable[[int, int], pd.DataFrame]

_CATEGORICAL_VALUES: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male"),
    "Partner": ("Yes", "No"),
    "Dependents": ("No", "Yes"),
    "PhoneService": ("Yes", "No"),
    "MultipleLines": ("No", "Yes", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": ("No", "Yes"),
    "OnlineBackup": ("Yes", "No"),
    "DeviceProtection": ("No", "Yes"),
    "TechSupport": ("No", "Yes"),
    "StreamingTV": ("No", "Yes"),
    "StreamingMovies": ("Yes", "No"),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": ("Yes", "No"),
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}


def synthetic_frame(n_no: int, n_yes: int) -> pd.DataFrame:
    """Build a contract-shaped frame: ``n_no`` + ``n_yes`` unique customers."""
    n = n_no + n_yes
    data: dict[str, object] = {
        ID_COLUMN: [f"CUST-{i:05d}" for i in range(n)],
        TARGET: ["No"] * n_no + ["Yes"] * n_yes,
        "tenure": [i % 72 for i in range(n)],
        "SeniorCitizen": [i % 2 for i in range(n)],
        "MonthlyCharges": [float((i * 7) % 150 + 20) for i in range(n)],
        # A few blank TotalCharges, as in the real dataset (zero-tenure rows).
        "TotalCharges": [
            float("nan") if i % 13 == 0 else float(i * 11) for i in range(n)
        ],
    }
    for spec in FEATURES:
        if spec.name in data:
            continue
        values = _CATEGORICAL_VALUES[spec.name]
        data[spec.name] = [values[i % len(values)] for i in range(n)]
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def make_frame() -> FrameFactory:
    return synthetic_frame
