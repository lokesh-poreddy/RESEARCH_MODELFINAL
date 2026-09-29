"""researchforge/policy/transfer_gate.py — Adaptive memory-transfer gate.

RF-2.0 Phase 5: Replaces the binary 0.5 failure-halving in policy_learner.py
with a continuous, learned gate that decides how much influence past memory
should have on the current action selection.

Scientific Rationale:
  The RF-1 system applies `score *= 0.5` when `has_similar_failure()` returns
  True. This is a hard binary gate that:
  1. Treats all failures equally (ignoring severity, recency, and relevance)
  2. Never recovers — once flagged, an action is permanently penalized
  3. Ignores positive evidence that may override stale negative evidence

  The TransferGate computes a continuous modifier g(a, c) ∈ [g_min, 1.0]:

    g(a, c) = sigmoid(
      w_pos · p_success(a, c)         # Posterior success probability
    - w_neg · p_failure_recent(a, c)   # Recent failure density
    + w_evi · log(1 + n_evidence)      # Evidence volume (more evidence → more trust)
    - w_unc · uncertainty(a, c)        # Posterior uncertainty (uncertain → less trust)
    + w_rec · recency_discount(a, c)   # Recency of last success
    + bias
    )

  Crucially:
  - The gate output is always ∈ [g_min, 1.0], never hard 0 or hard 1
  - The gate parameters can be learned from trajectory data (Phase 7)
  - The gate provides a traceable decomposition for every decision

PRESERVED: The binary halving in PolicyLearner is NOT modified. This module
provides a parallel mechanism used only by the new contextual policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..memory.posterior import PosteriorQueryResult


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


@dataclass
class GateWeights:
    """Weights for the transfer gate linear scoring function.

    These can be:
    1. Set manually (interpretable defaults)
    2. Learned from trajectory data via simple gradient descent
    3. Cross-validated on held-out generation subsequences
    """
    w_success: float = 2.0        # Weight on posterior success probability
    w_failure: float = -3.0       # Weight on recent failure density (negative: penalizes failure)
    w_evidence: float = 0.5       # Weight on log evidence volume
    w_uncertainty: float = -1.5   # Weight on posterior uncertainty (negative: penalizes uncertainty)
    w_recency: float = 0.8        # Weight on recency of last success
    bias: float = 0.0             # Learnable bias

    def to_dict(self) -> Dict[str, float]:
        return {
            "w_success": self.w_success,
            "w_failure": self.w_failure,
            "w_evidence": self.w_evidence,
            "w_uncertainty": self.w_uncertainty,
            "w_recency": self.w_recency,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "GateWeights":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GateDecision:
    """Traceable record of a single gate computation."""
    action: str
    gate_value: float              # g(a,c) in [g_min, 1.0]
    raw_logit: float               # Pre-sigmoid linear score
    components: Dict[str, float]   # Decomposed score components
    posterior_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "gate_value": round(self.gate_value, 6),
            "raw_logit": round(self.raw_logit, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()},
        }


class TransferGate:
    """Adaptive memory-transfer gate.

    Computes a continuous modifier g(a, c) ∈ [g_min, 1.0] for each action
    based on posterior memory evidence and recent trajectory signals.

    Parameters
    ----------
    weights : GateWeights
        Linear scoring weights. Can be learned or set manually.
    g_min : float
        Minimum gate value. Prevents any action from being completely
        suppressed by memory. Default 0.15 (matching the 15% floor
        from the paper's transfer control section).
    """

    def __init__(self, weights: Optional[GateWeights] = None,
                 g_min: float = 0.15) -> None:
        if g_min < 0.0 or g_min > 1.0:
            raise ValueError(f"g_min must be in [0, 1], got {g_min}")

        self.weights = weights or GateWeights()
        self.g_min = g_min

    def compute(
        self,
        action: str,
        posterior: PosteriorQueryResult,
        recent_failure_rate: float = 0.0,
        recency_score: float = 0.5,
    ) -> GateDecision:
        """Compute the transfer gate value for a single action.

        Args:
            action: Strategy name.
            posterior: PosteriorQueryResult from PosteriorMemory.query().
            recent_failure_rate: Fraction of recent trials that failed
                for this action in the current context [0, 1].
            recency_score: How recently this action last succeeded [0, 1].
                1.0 = very recent, 0.0 = never or long ago.

        Returns:
            GateDecision with full decomposition.
        """
        w = self.weights

        # Compute linear score components
        comp_success = w.w_success * posterior.success_probability
        comp_failure = w.w_failure * recent_failure_rate
        comp_evidence = w.w_evidence * math.log(1.0 + sum(posterior.evidence_counts.values()))
        comp_uncertainty = w.w_uncertainty * posterior.uncertainty
        comp_recency = w.w_recency * recency_score

        raw_logit = (comp_success + comp_failure + comp_evidence +
                     comp_uncertainty + comp_recency + w.bias)

        # Map through sigmoid and scale to [g_min, 1.0]
        sig = _sigmoid(raw_logit)
        gate_value = self.g_min + (1.0 - self.g_min) * sig

        return GateDecision(
            action=action,
            gate_value=gate_value,
            raw_logit=raw_logit,
            components={
                "success": comp_success,
                "failure": comp_failure,
                "evidence": comp_evidence,
                "uncertainty": comp_uncertainty,
                "recency": comp_recency,
                "bias": w.bias,
            },
            posterior_summary=posterior.to_dict(),
        )

    def compute_all(
        self,
        actions: list[str],
        posteriors: Dict[str, PosteriorQueryResult],
        recent_failure_rates: Optional[Dict[str, float]] = None,
        recency_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, GateDecision]:
        """Compute gate values for all actions.

        Args:
            actions: List of action names.
            posteriors: Dict mapping action → PosteriorQueryResult.
            recent_failure_rates: Optional dict mapping action → failure rate.
            recency_scores: Optional dict mapping action → recency score.

        Returns:
            Dict mapping action → GateDecision.
        """
        results: Dict[str, GateDecision] = {}
        for a in actions:
            posterior = posteriors.get(a)
            if posterior is None:
                # Create a maximally uncertain prior-only result
                posterior = PosteriorQueryResult(
                    success_probability=0.5,
                    uncertainty=1.0,
                    confidence=0.0,
                    mean_delta=0.0,
                    primary_level=0,
                    evidence_counts={0: 0, 1: 0, 2: 0, 3: 0},
                )

            failure_rate = (recent_failure_rates or {}).get(a, 0.0)
            recency = (recency_scores or {}).get(a, 0.5)

            results[a] = self.compute(a, posterior, failure_rate, recency)

        return results

    def modifier_fn(
        self,
        posteriors: Dict[str, PosteriorQueryResult],
        recent_failure_rates: Optional[Dict[str, float]] = None,
        recency_scores: Optional[Dict[str, float]] = None,
    ) -> callable:
        """Return a callable(action) → float for use as the policy's
        memory_modifier argument.

        This bridges the TransferGate to the ContextualBanditPolicy.select_action()
        API, providing backward compatibility with the modifier pattern.
        """
        # Pre-compute all gate values
        gate_decisions = self.compute_all(
            list(posteriors.keys()),
            posteriors,
            recent_failure_rates,
            recency_scores,
        )
        return lambda a: gate_decisions.get(a, GateDecision(
            action=a, gate_value=1.0, raw_logit=0.0, components={}
        )).gate_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_type": "TransferGate",
            "g_min": self.g_min,
            "weights": self.weights.to_dict(),
        }
