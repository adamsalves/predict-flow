"""Candidate set: baseline present, unique names, buildable pipelines."""

from __future__ import annotations

from training.candidates import BASELINE_NAME, default_candidates
from training.features import FEATURES, TARGET
from training.preprocessing import build_pipeline
from training.tests.conftest import FrameFactory


def test_baseline_is_first_and_is_lr() -> None:
    candidates = default_candidates()
    assert candidates[0].name == BASELINE_NAME
    assert candidates[0].model == "lr"


def test_candidate_names_are_unique() -> None:
    names = [c.name for c in default_candidates()]
    assert len(names) == len(set(names))


def test_every_candidate_builds_and_fits(make_frame: FrameFactory) -> None:
    df = make_frame(30, 10)
    features = df[[spec.name for spec in FEATURES]]
    y = df[TARGET].map({"No": 0.0, "Yes": 1.0})
    for candidate in default_candidates():
        pipeline = build_pipeline(candidate.model, candidate.hyperparameters, seed=42)
        pipeline.fit(features, y)
        proba = pipeline.predict_proba(features.head(2))
        assert proba.shape == (2, 2)
