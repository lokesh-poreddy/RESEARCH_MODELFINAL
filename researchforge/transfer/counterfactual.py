"""researchforge/transfer/counterfactual.py — Counterfactual Evidence Service.

Phase 12A:
Provides epistemic evidence (OBSERVED, ESTIMATED, UNAVAILABLE) about transfer utility.
This service does NOT make operational transfer decisions. It purely supplies the
measured or predicted counterfactual $\\Delta$ to the downstream Utility Learner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple


class CounterfactualStatus(str, Enum):
    """Epistemic status of a counterfactual quality evaluation."""
    OBSERVED = "OBSERVED"      # Both prior and fresh paths empirically evaluated
    ESTIMATED = "ESTIMATED"    # Validated surrogate model prediction with variance
    UNAVAILABLE = "UNAVAILABLE" # No paired runs or surrogate model; cannot fabricate


@dataclass(frozen=True)
class TransferEvidence:
    """Epistemic evidence of counterfactual analysis for a candidate transfer."""
    status: CounterfactualStatus
    expected_delta: Optional[float]
    variance: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CounterfactualEvidenceService:
    """Provides evidence about the utility of a transfer decision without gating it."""

    def __init__(
        self,
        utility_predictor: Optional[Callable[[Dict[str, Any]], Tuple[float, float]]] = None,
    ) -> None:
        # The predictor should ideally be the Phase 12B Transfer Utility Learner.
        self.utility_predictor = utility_predictor

    def get_evidence(
        self,
        context_features: Dict[str, Any],
        paired_evidence: Optional[Dict[str, float]] = None,
    ) -> TransferEvidence:
        """Fetch counterfactual evidence for a transfer given the context.
        
        Args:
            context_features: The full Phase 12 context vector (Cs, Ct, Cst, M, a).
            paired_evidence: Optional dict with 'quality_prior' and 'quality_fresh'.
        """
        # 1. Case: OBSERVED (We actually ran a paired counterfactual experiment)
        if paired_evidence and "quality_prior" in paired_evidence and "quality_fresh" in paired_evidence:
            q_prior = paired_evidence["quality_prior"]
            q_fresh = paired_evidence["quality_fresh"]
            obs_delta = q_prior - q_fresh
            return TransferEvidence(
                status=CounterfactualStatus.OBSERVED,
                expected_delta=obs_delta,
                variance=0.0,
                confidence_interval=(obs_delta, obs_delta),
                rationale=f"Observed paired execution: Q(prior)={q_prior:.4f}, Q(fresh)={q_fresh:.4f}, Delta={obs_delta:+.4f}.",
            )

        # 2. Case: ESTIMATED (Predictor supplies E[Delta] and Var[Delta])
        if self.utility_predictor is not None:
            try:
                pred_mean, pred_var = self.utility_predictor(context_features)
                ci_half = 1.96 * (pred_var ** 0.5)
                ci = (pred_mean - ci_half, pred_mean + ci_half)
                return TransferEvidence(
                    status=CounterfactualStatus.ESTIMATED,
                    expected_delta=pred_mean,
                    variance=pred_var,
                    confidence_interval=ci,
                    rationale=f"Estimated by utility predictor: E[Delta]={pred_mean:+.4f}, Var={pred_var:.4f}.",
                )
            except Exception as e:
                return TransferEvidence(
                    status=CounterfactualStatus.UNAVAILABLE,
                    expected_delta=None,
                    variance=None,
                    confidence_interval=None,
                    rationale=f"Utility predictor failed: {e}.",
                )

        # 3. Case: UNAVAILABLE
        return TransferEvidence(
            status=CounterfactualStatus.UNAVAILABLE,
            expected_delta=None,
            variance=None,
            confidence_interval=None,
            rationale="No paired observation and no utility predictor available.",
        )
