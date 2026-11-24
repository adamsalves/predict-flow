"""Gate semantics: clear win, practical tie, calibration veto, fallback."""

from __future__ import annotations

import pytest

from training.candidates import BASELINE_NAME
from training.metrics import MetricWithCI, SplitMetrics
from training.selection import (
    CandidateEvaluation,
    decide,
)


def _metrics(
    pr_auc: tuple[float, float, float],
    brier: tuple[float, float, float],
) -> SplitMetrics:
    return SplitMetrics(
        roc_auc=MetricWithCI(value=0.8, ci_low=0.78, ci_high=0.82),
        pr_auc=MetricWithCI(value=pr_auc[0], ci_low=pr_auc[1], ci_high=pr_auc[2]),
        brier=MetricWithCI(value=brier[0], ci_low=brier[1], ci_high=brier[2]),
        base_rate=0.26,
        calibration=(),
    )


def _candidate(
    name: str,
    pr_auc: tuple[float, float, float],
    brier: tuple[float, float, float] = (0.14, 0.13, 0.15),
) -> CandidateEvaluation:
    return CandidateEvaluation(
        name=name,
        model="lr" if name == BASELINE_NAME else "rf",
        hyperparameters={},
        val_metrics=_metrics(pr_auc, brier),
    )


_BASELINE = _candidate(BASELINE_NAME, (0.64, 0.60, 0.68))


def test_clear_win_promotes_challenger() -> None:
    challenger = _candidate("rf-deep", (0.72, 0.69, 0.75))
    decision = decide((_BASELINE, challenger))
    assert decision.winner == "rf-deep"
    assert decision.promoted is True
    assert decision.comparisons[0].pr_auc_clear_win is True
    assert decision.comparisons[0].brier_veto is False


def test_overlapping_cis_are_a_practical_tie_and_keep_baseline() -> None:
    challenger = _candidate("rf-deep", (0.66, 0.62, 0.70))
    decision = decide((_BASELINE, challenger))
    assert decision.winner == BASELINE_NAME
    assert decision.promoted is False
    assert decision.comparisons[0].pr_auc_clear_win is False


def test_clear_loss_keeps_baseline() -> None:
    challenger = _candidate("rf-deep", (0.60, 0.56, 0.63))
    decision = decide((_BASELINE, challenger))
    assert decision.winner == BASELINE_NAME
    assert decision.promoted is False


def test_brier_veto_blocks_promotion_despite_pr_auc_win() -> None:
    challenger = _candidate(
        "rf-deep", (0.72, 0.69, 0.75), brier=(0.18, 0.17, 0.19)
    )
    decision = decide((_BASELINE, challenger))
    assert decision.winner == BASELINE_NAME
    assert decision.promoted is False
    assert decision.comparisons[0].pr_auc_clear_win is True
    assert decision.comparisons[0].brier_veto is True


def test_best_pr_auc_wins_among_promoted_challengers() -> None:
    weak = _candidate("rf-a", (0.70, 0.69, 0.71))
    strong = _candidate("rf-b", (0.74, 0.72, 0.76))
    decision = decide((_BASELINE, weak, strong))
    assert decision.winner == "rf-b"
    assert {c.challenger for c in decision.comparisons} == {"rf-a", "rf-b"}


def test_missing_baseline_raises() -> None:
    with pytest.raises(ValueError, match="baseline"):
        decide((_candidate("rf-deep", (0.72, 0.69, 0.75)),))


def test_duplicate_candidate_names_raise() -> None:
    first = _candidate("rf-deep", (0.72, 0.69, 0.75))
    second = _candidate("rf-deep", (0.60, 0.56, 0.63))
    with pytest.raises(ValueError, match="unique"):
        decide((_BASELINE, first, second))


def test_decision_serializes_gate_evidence() -> None:
    challenger = _candidate("rf-deep", (0.72, 0.69, 0.75))
    decision = decide((_BASELINE, challenger))
    block = decision.to_dict((_BASELINE, challenger))
    assert block["winner"] == "rf-deep"
    assert block["promoted"] is True
    assert block["gate_rule"]
    assert block["limitations"]
    candidates = block["candidates"]
    assert isinstance(candidates, list)
    assert {c["name"] for c in candidates} == {BASELINE_NAME, "rf-deep"}
    comparisons = block["comparisons"]
    assert isinstance(comparisons, list)
    assert comparisons[0]["pr_auc_clear_win"] is True
