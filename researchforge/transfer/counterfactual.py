"""researchforge/transfer/counterfactual.py — Counterfactual Evidence & Arbitration Service.

Phase 12A & Phase 13:
Provides epistemic evidence (COUNTERFACTUAL_OBSERVED, COUNTERFACTUAL_ESTIMATED,
COUNTERFACTUAL_UNAVAILABLE) about transfer utility.
Supports both the downstream Utility Learner pipeline and explicit
tri-state counterfactual arbitration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple


class CounterfactualStatus(str, Enum):
    """Epistemic status of a counterfactual quality evaluation."""
    COUNTERFACTUAL_OBSERVED = "COUNTERFACTUAL_OBSERVED"
    COUNTERFACTUAL_ESTIMATED = "COUNTERFACTUAL_ESTIMATED"
    COUNTERFACTUAL_UNAVAILABLE = "COUNTERFACTUAL_UNAVAILABLE"

    # Compatibility aliases
    OBSERVED = "COUNTERFACTUAL_OBSERVED"
    ESTIMATED = "COUNTERFACTUAL_ESTIMATED"
    UNAVAILABLE = "COUNTERFACTUAL_UNAVAILABLE"


@dataclass(frozen=True)
class CounterfactualEvaluation:
    """Epistemic evaluation of counterfactual analysis for a candidate transfer."""
    status: CounterfactualStatus
    expected_delta: Optional[float]
    variance: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    recommendation: str
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# TransferEvidence alias for backwards-compatibility with existing pipelines
TransferEvidence = CounterfactualEvaluation


class CounterfactualArbitrator:
    """Tri-state counterfactual arbitrator evaluating candidate transfers."""

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
        """Arbitrate transfer candidate using paired execution, surrogate model, or fallback."""
        # 1. Case: COUNTERFACTUAL_OBSERVED
        if (
            paired_evidence is not None
            and "quality_prior" in paired_evidence
            and "quality_fresh" in paired_evidence
        ):
            q_prior = paired_evidence["quality_prior"]
            q_fresh = paired_evidence["quality_fresh"]
            obs_delta = q_prior - q_fresh
            rec = "PROCEED_WITH_TRANSFER" if obs_delta >= 0.0 else "SUPPRESS_TRANSFER"
            return CounterfactualEvaluation(
                status=CounterfactualStatus.COUNTERFACTUAL_OBSERVED,
                expected_delta=obs_delta,
                variance=0.0,
                confidence_interval=(obs_delta, obs_delta),
                recommendation=rec,
                rationale=f"Observed paired execution: Q(prior)={q_prior:.4f}, Q(fresh)={q_fresh:.4f}, Delta={obs_delta:+.4f}.",
            )

        # 2. Case: COUNTERFACTUAL_ESTIMATED
        if self.surrogate_model is not None:
            try:
                pred_mean, pred_var = self.surrogate_model(candidate_spec)
                ci_half = 1.96 * (pred_var ** 0.5)
                ci = (pred_mean - ci_half, pred_mean + ci_half)
                rec = "PROCEED_WITH_TRANSFER" if pred_mean >= 0.0 else "SUPPRESS_TRANSFER"
                return CounterfactualEvaluation(
                    status=CounterfactualStatus.COUNTERFACTUAL_ESTIMATED,
                    expected_delta=pred_mean,
                    variance=pred_var,
                    confidence_interval=ci,
                    recommendation=rec,
                    rationale=f"Estimated by surrogate model: E[Delta]={pred_mean:+.4f}, Var={pred_var:.4f}.",
                )
            except Exception as e:
                pass

        # 3. Case: COUNTERFACTUAL_UNAVAILABLE
        return CounterfactualEvaluation(
            status=CounterfactualStatus.COUNTERFACTUAL_UNAVAILABLE,
            expected_delta=None,
            variance=None,
            confidence_interval=None,
            recommendation="FALLBACK_TO_EVIDENCE_GATING",
            rationale="Counterfactual quality cannot be known without empirical paired execution or validated surrogate model.",
        )


class CounterfactualEvidenceService:
    """Provides evidence about the utility of a transfer decision without gating it."""

    def __init__(
        self,
        utility_predictor: Optional[Callable[[Dict[str, Any]], Tuple[float, float]]] = None,
    ) -> None:
        self.utility_predictor = utility_predictor
        self._arbitrator = CounterfactualArbitrator(surrogate_model=utility_predictor)

    def get_evidence(
        self,
        context_features: Dict[str, Any],
        paired_evidence: Optional[Dict[str, float]] = None,
    ) -> TransferEvidence:
        """Fetch counterfactual evidence for a transfer given the context."""
        return self._arbitrator.arbitrate(
            candidate_spec=context_features,
            paired_evidence=paired_evidence,
        )
