"""researchforge/policy/config.py — ResearchPolicy configuration contract.

Scientific & Architectural Constraints (Phase 9):
The default weights defined here represent an inspectable heuristic baseline (labeled
explicitly as 'baseline_v1' / 'heuristic_prior'). They are NOT scientifically optimized
or learned constants. All weights are transparent, versioned, and content-addressed.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ..domain.base import DomainObject, _canonical_json


class PolicyAblationMode(str, Enum):
    """Controlled policy ablation conditions sharing identical tasks, budget, and evaluations."""
    RANDOM = "RANDOM"
    STATIC_BASELINE = "STATIC_BASELINE"
    UCB = "UCB"
    MEMORY_CONDITIONED = "MEMORY_CONDITIONED"
    ADAPTIVE_POLICY = "ADAPTIVE_POLICY"


@dataclass(frozen=True)
class PolicyConfig(DomainObject):
    """Immutable, versioned policy configuration for transparent action evaluation.
    
    Weights are heuristic baseline priors ('baseline_v1') and must not be claimed
    as empirically optimal.
    """
    id: str = "policy_cfg_baseline_v1"
    schema_version: str = "1.0"
    policy_version: str = "baseline_v1"
    config_label: str = "heuristic_prior"
    ablation_mode: PolicyAblationMode = PolicyAblationMode.ADAPTIVE_POLICY

    # Inspectable component weights (heuristic baseline prior)
    w_performance: float = 1.0
    w_information_gain: float = 0.8
    w_novelty: float = 0.5
    w_evidence: float = 0.7
    w_transfer: float = 0.6
    w_failure_risk: float = 1.2
    w_redundancy: float = 0.8
    w_cost: float = 0.3

    min_confidence: float = 0.1
    max_candidates: int = 20
    provenance_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def weights_dict(self) -> Dict[str, float]:
        return {
            "w_performance": self.w_performance,
            "w_information_gain": self.w_information_gain,
            "w_novelty": self.w_novelty,
            "w_evidence": self.w_evidence,
            "w_transfer": self.w_transfer,
            "w_failure_risk": self.w_failure_risk,
            "w_redundancy": self.w_redundancy,
            "w_cost": self.w_cost,
        }

    @classmethod
    def baseline_v1(cls, ablation_mode: PolicyAblationMode = PolicyAblationMode.ADAPTIVE_POLICY) -> "PolicyConfig":
        """Factory for the canonical Phase 9 baseline heuristic prior."""
        return cls(
            id=f"policy_cfg_{ablation_mode.value.lower()}_v1",
            policy_version="baseline_v1",
            config_label="heuristic_prior",
            ablation_mode=ablation_mode,
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PolicyConfig":
        d = dict(obj)
        if "ablation_mode" in d and isinstance(d["ablation_mode"], str):
            d["ablation_mode"] = PolicyAblationMode(d["ablation_mode"])
        return cls(**d)
