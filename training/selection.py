"""Validation gate for model selection (ADR-0004, ADR-0009).

Candidates are compared on the validation split only; the test split is
touched exactly once, after the winner, its calibration and its threshold
are frozen. LogisticRegression is the mandatory baseline: a challenger is
promoted only on a clear win, and any practical tie or defeat keeps the
baseline, which is always trained and available as fallback.

Gate rule (fixed before training, recorded in the manifest):

- primary metric: validation PR-AUC. A challenger wins clearly only when
  its 95% bootstrap CI lies entirely above the baseline's (non-overlapping
  intervals); overlapping CIs are a practical tie and keep the baseline.
- complementary criterion: the challenger's calibrated Brier score must
  not be clearly worse (its CI entirely above the baseline's vetoes
  promotion), so a discrimination gain never ships a miscalibrated model.
"""

from __future__ import annotations

from dataclasses import dataclass

from training.candidates import BASELINE_NAME
from training.metrics import SplitMetrics

GATE_RULE = (
    "primary: validation PR-AUC; a challenger is promoted only on a clear "
    "win (its 95% bootstrap CI lower bound above the baseline CI upper "
    "bound); overlapping CIs are a practical tie and keep the baseline. "
    "complementary: calibrated Brier veto — a challenger whose Brier CI "
    "lower bound is above the baseline CI upper bound is not promoted even "
    "on a PR-AUC clear win."
)
SELECTION_LIMITATIONS = (
    "selection uses the validation split only; the test split is evaluated "
    "once after the winner, calibration and threshold are frozen. Bootstrap "
    "CIs are Monte Carlo estimates on a historical dataset with no "
    "prospective horizon (see target.limitations); a gate win is evidence "
    "of better historical discrimination, not of prospective performance."
)


@dataclass(frozen=True)
class CandidateEvaluation:
    """A candidate's frozen configuration and its validation metrics."""

    name: str
    model: str
    hyperparameters: dict[str, object]
    val_metrics: SplitMetrics


@dataclass(frozen=True)
class ChallengerComparison:
    challenger: str
    pr_auc_clear_win: bool
    brier_veto: bool
    promoted: bool


@dataclass(frozen=True)
class GateDecision:
    winner: str
    promoted: bool
    reason: str
    comparisons: tuple[ChallengerComparison, ...]

    def to_dict(
        self, evaluations: tuple[CandidateEvaluation, ...]
    ) -> dict[str, object]:
        by_name = {e.name: e for e in evaluations}
        return {
            "gate_rule": GATE_RULE,
            "limitations": SELECTION_LIMITATIONS,
            "winner": self.winner,
            "promoted": self.promoted,
            "reason": self.reason,
            "candidates": [
                {
                    "name": e.name,
                    "model": e.model,
                    "hyperparameters": e.hyperparameters,
                    "validation": _metric_summary(e.val_metrics),
                }
                for e in evaluations
            ],
            "comparisons": [
                {
                    "challenger": c.challenger,
                    "pr_auc_clear_win": c.pr_auc_clear_win,
                    "brier_veto": c.brier_veto,
                    "promoted": c.promoted,
                }
                for c in self.comparisons
            ],
            "baseline": BASELINE_NAME,
            "winner_model": by_name[self.winner].model,
        }


def _metric_summary(metrics: SplitMetrics) -> dict[str, object]:
    return {
        "pr_auc": metrics.pr_auc.value,
        "pr_auc_ci95": [metrics.pr_auc.ci_low, metrics.pr_auc.ci_high],
        "roc_auc": metrics.roc_auc.value,
        "roc_auc_ci95": [metrics.roc_auc.ci_low, metrics.roc_auc.ci_high],
        "brier": metrics.brier.value,
        "brier_ci95": [metrics.brier.ci_low, metrics.brier.ci_high],
        "base_rate": metrics.base_rate,
    }


def decide(
    evaluations: tuple[CandidateEvaluation, ...],
    baseline_name: str = BASELINE_NAME,
) -> GateDecision:
    """Apply the gate to the validation evaluations and pick the winner."""
    by_name = {e.name: e for e in evaluations}
    if baseline_name not in by_name:
        raise ValueError(f"baseline candidate {baseline_name!r} is required")
    baseline = by_name[baseline_name]

    comparisons: list[ChallengerComparison] = []
    winners: list[CandidateEvaluation] = []
    for candidate in evaluations:
        if candidate.name == baseline_name:
            continue
        clear_win = (
            candidate.val_metrics.pr_auc.ci_low
            > baseline.val_metrics.pr_auc.ci_high
        )
        brier_veto = (
            candidate.val_metrics.brier.ci_low > baseline.val_metrics.brier.ci_high
        )
        promoted = clear_win and not brier_veto
        comparisons.append(
            ChallengerComparison(
                challenger=candidate.name,
                pr_auc_clear_win=clear_win,
                brier_veto=brier_veto,
                promoted=promoted,
            )
        )
        if promoted:
            winners.append(candidate)

    if not winners:
        reason = (
            "no challenger beat the baseline clearly on validation PR-AUC "
            "without a calibration veto; practical tie or defeat keeps the "
            "simpler model (ADR-0004)"
        )
        return GateDecision(
            winner=baseline_name,
            promoted=False,
            reason=reason,
            comparisons=tuple(comparisons),
        )

    best = max(winners, key=lambda e: (e.val_metrics.pr_auc.value, e.name))
    reason = (
        f"{best.name} won the validation gate on PR-AUC "
        f"({best.val_metrics.pr_auc.value:.4f} vs baseline "
        f"{baseline.val_metrics.pr_auc.value:.4f}, non-overlapping 95% CIs) "
        "without calibrated-Brier veto"
    )
    return GateDecision(
        winner=best.name,
        promoted=True,
        reason=reason,
        comparisons=tuple(comparisons),
    )
