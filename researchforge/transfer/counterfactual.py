"""researchforge/transfer/counterfactual.py — Tri-state counterfactual arbitration.

Phase 13 (RF-1.0.0-beta.1):
Explicitly models COUNTERFACTUAL_OBSERVED, COUNTERFACTUAL_ESTIMATED, and
COUNTERFACTUAL_UNAVAILABLE. When unavailable, the system never fabricates
expectations; it falls back to TransferGuard's evidence-based policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class CounterfactualStatus(str, Enum):
    """Epistemic status of a counterfactual quality evaluation."""
    COUNTERFACTUAL_OBSERVED = "COUNTERFACTUAL_OBSERVED"      # Both prior and fresh paths empirically evaluated
    COUNTERFACTUAL_ESTIMATED = "COUNTERFACTUAL_ESTIMATED"    # Validated surrogate model prediction with variance
    COUNTERFACTUAL_UNAVAILABLE = "COUNTERFACTUAL_UNAVAILABLE" # No paired runs or surrogate model; cannot fabricate


@dataclass(frozen=True)
class CounterfactualEvaluation:
    """Diagnostic outcome of counterfactual analysis for a candidate decision."""
    status: CounterfactualStatus
    expected_delta: Optional[float]
    variance: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    recommendation: str  # "PROCEED_WITH_TRANSFER" | "SUPPRESS_TRANSFER" | "FALLBACK_TO_EVIDENCE_GATING"
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CounterfactualArbitrator:
    """Evaluates whether transferring prior memory is expected to improve decision quality."""

    def __init__(
        self,
        surrogate_model: Optional[Callable[[Dict[str, Any]], Tuple[float, float]]] = None,
    ) -> None:
        self.surrogate_model = surrogate_model

    def arbitrate(
        self,
        candidate_spec: Dict[str, Any],
        paired_evidence: Optional[Dict[str, float]] = None,
    ) -> CounterfactualEvaluation:
        """Arbitrate decision based on counterfactual evidence.
        
        Args:
            candidate_spec: Specification of the candidate genome/action.
            paired_evidence: Optional dict with 'quality_prior' and 'quality_fresh' if paired run occurred.
        """
        # 1. Case: COUNTERFACTUAL_OBSERVED
        if paired_evidence and "quality_prior" in paired_evidence and "quality_fresh" in paired_evidence:
            q_prior = paired_evidence["quality_prior"]
            q_fresh = paired_evidence["quality_fresh"]
            obs_delta = q_prior - q_fresh
            rec = "PROCEED_WITH_TRANSFER" if obs_delta >= 0 else "SUPPRESS_TRANSFER"
            rat = (
                f"Observed paired execution: Q(prior)={q_prior:.4f}, Q(fresh)={q_fresh:.4f}, "
                f"Delta={obs_delta:+.4f}."
            )
            return CounterfactualEvaluation(
                status=CounterfactualStatus.COUNTERFACTUAL_OBSERVED,
                expected_delta=obs_delta,
                variance=0.0,
                confidence_interval=(obs_delta, obs_delta),
                recommendation=rec,
                rationale=rat,
            )

        # 2. Case: COUNTERFACTUAL_ESTIMATED (Surrogate prediction)
        if self.surrogate_model is not None:
            try:
                pred_mean, pred_var = self.surrogate_model(candidate_spec)
                ci_half = 1.96 * (pred_var ** 0.5)
                ci = (pred_mean - ci_half, pred_mean + ci_half)
                rec = "PROCEED_WITH_TRANSFER" if pred_mean >= 0 else "SUPPRESS_TRANSFER"
                rat = (
                    f"Estimated by surrogate model: E[Delta]={pred_mean:+.4f}, "
                    f"Var={pred_var:.4f}, 95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]."
                )
                return CounterfactualEvaluation(
                    status=CounterfactualStatus.COUNTERFACTUAL_ESTIMATED,
                    expected_delta=pred_mean,
                    variance=pred_var,
                    confidence_interval=ci,
                    recommendation=rec,
                    rationale=rat,
                )
            except Exception as e:
                # Surrogate failure falls back to unavailable
                return CounterfactualEvaluation(
                    status=CounterfactualStatus.COUNTERFACTUAL_UNAVAILABLE,
                    expected_delta=None,
                    variance=None,
                    confidence_interval=None,
                    recommendation="FALLBACK_TO_EVIDENCE_GATING",
                    rationale=f"Surrogate model estimation failed: {e}. Falling back to evidence gating.",
                )

        # 3. Case: COUNTERFACTUAL_UNAVAILABLE
        return CounterfactualEvaluation(
            status=CounterfactualStatus.COUNTERFACTUAL_UNAVAILABLE,
            expected_delta=None,
            variance=None,
            confidence_interval=None,
            recommendation="FALLBACK_TO_EVIDENCE_GATING",
            rationale=(
                "No paired observation and no surrogate model available. "
                "Counterfactual expectation cannot be known; falling back cleanly to TransferGuard evidence gating."
            ),
        )
