"""researchforge/policy/utility_gate.py — Phase 12D Utility-Aware Gate.

Consumes probabilistic transfer utility (P_B, P_N, P_H) from the Utility Predictor
and outputs discrete policy actions (TRANSFER, ATTENUATE, SUPPRESS, ABSTAIN)
and a continuous UCB modifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class GateAction(str, Enum):
    """Discrete operational decision for a transfer strategy."""
    TRANSFER = "TRANSFER"    # Positively weight the evidence
    ATTENUATE = "ATTENUATE"  # Heavily decay or neutralize the evidence
    SUPPRESS = "SUPPRESS"    # Actively penalize the strategy
    ABSTAIN = "ABSTAIN"      # High uncertainty; defer to counterfactual sampling


@dataclass(frozen=True)
class UtilityGateDecision:
    """Traceable record of a utility-aware gate computation."""
    action: str
    gate_action: GateAction
    modifier: float  # The continuous adjustment applied to the action's Q-value
    p_benefit: float
    p_neutral: float
    p_harm: float
    expected_delta: float
    uncertainty: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "gate_action": self.gate_action.value,
            "modifier": round(self.modifier, 6),
            "p_benefit": round(self.p_benefit, 4),
            "p_neutral": round(self.p_neutral, 4),
            "p_harm": round(self.p_harm, 4),
            "expected_delta": round(self.expected_delta, 6),
            "uncertainty": round(self.uncertainty, 6),
        }


class UtilityAwareGate:
    """Utility-Aware Gate for Contextual Bandits.
    
    Implements:
    g_{transfer} = lambda_B * P_B - lambda_H * P_H
    with an abstention fallback for high uncertainty.
    """

    def __init__(
        self,
        lambda_b: float = 1.0,
        lambda_h: float = 2.0,  # Asymmetric penalty for negative transfer
        tau_confidence: float = 0.4, # Minimum probability mass required to act
    ) -> None:
        self.lambda_b = lambda_b
        self.lambda_h = lambda_h
        self.tau_confidence = tau_confidence

    def compute(
        self,
        action: str,
        p_b: float,
        p_n: float,
        p_h: float,
        expected_delta: float,
        uncertainty: float,
    ) -> UtilityGateDecision:
        """Computes the operational gate decision based on predictive utility."""
        
        # 1. Uncertainty Check -> ABSTAIN
        # If the model strongly believes nothing (all probabilities are uniform/low)
        if max(p_b, p_n, p_h) < self.tau_confidence:
            gate_action = GateAction.ABSTAIN
            modifier = 0.0
            
        # 2. Harm Check -> SUPPRESS
        # If Harm is the most likely outcome
        elif p_h >= p_b and p_h >= p_n:
            gate_action = GateAction.SUPPRESS
            modifier = -self.lambda_h * p_h
            
        # 3. Benefit Check -> TRANSFER
        # If Benefit is the most likely outcome
        elif p_b > p_h and p_b > p_n:
            gate_action = GateAction.TRANSFER
            modifier = self.lambda_b * p_b
            
        # 4. Neutral Check -> ATTENUATE
        # If Neutral is the most likely outcome
        else:
            gate_action = GateAction.ATTENUATE
            # Neutral implies Delta ~ 0. We don't boost, we slightly decay or ignore.
            modifier = 0.0

        return UtilityGateDecision(
            action=action,
            gate_action=gate_action,
            modifier=modifier,
            p_benefit=p_b,
            p_neutral=p_n,
            p_harm=p_h,
            expected_delta=expected_delta,
            uncertainty=uncertainty,
        )
