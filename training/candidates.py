"""Model candidates for the validation gate (ADR-0004).

LogisticRegression is the mandatory baseline and the permanent fallback;
every RandomForest configuration is a challenger that must win the
validation gate to be promoted. The candidate set is deliberately small —
hyperparameters are selected on the validation split, never on the test
split (ADR-0009) — and fixed before training so the comparison is
auditable from the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ModelKind = Literal["lr", "rf"]

BASELINE_NAME = "lr-baseline"


@dataclass(frozen=True)
class CandidateSpec:
    """One model configuration entering the validation gate."""

    name: str
    model: ModelKind
    hyperparameters: dict[str, object] = field(default_factory=dict)


def default_candidates() -> tuple[CandidateSpec, ...]:
    """Baseline first, then the RF challenger configurations."""
    return (
        CandidateSpec(
            name=BASELINE_NAME,
            model="lr",
            hyperparameters={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        ),
        CandidateSpec(
            name="rf-deep",
            model="rf",
            hyperparameters={
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 5,
                "class_weight": "balanced_subsample",
            },
        ),
        CandidateSpec(
            name="rf-regularized",
            model="rf",
            hyperparameters={
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_leaf": 20,
                "class_weight": "balanced_subsample",
            },
        ),
    )
