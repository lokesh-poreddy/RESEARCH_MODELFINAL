"""researchforge/policy/score_decomposition.py — Transparent utility score decomposition.

Scientific & Auditability Requirements (Phase 9):
1. Keep both raw component values and weighted contributions visible.
2. Transfer must be explicitly classified (POSITIVE_TRANSFER, NEGATIVE_TRANSFER, etc.).
3. Generate deterministic human-readable explanations answering 'Why was this action selected?'
   directly from components, without post-hoc black-box generation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional

from ..domain.base import DomainObject


class TransferAssessment(str, Enum):
    """Categorical classification of cross-context transfer potential."""
    POSITIVE_TRANSFER = "POSITIVE_TRANSFER"
    NEGATIVE_TRANSFER = "NEGATIVE_TRANSFER"
    UNCERTAIN_TRANSFER = "UNCERTAIN_TRANSFER"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class ScoreDecomposition(DomainObject):
    """Inspectable score decomposition for a candidate ResearchAction."""
    id: str
    schema_version: str
    action_id: str
    action_type: str

    # Raw component scores (typically normalized in [0.0, 1.0])
    performance: float = 0.0
    information_gain: float = 0.0
    novelty: float = 0.0
    evidence: float = 0.0
    transfer: float = 0.0
    failure_risk: float = 0.0
    redundancy: float = 0.0
    cost: float = 0.0

    # Weighted contributions: (weight * raw_value)
    w_performance: float = 0.0
    w_information_gain: float = 0.0
    w_novelty: float = 0.0
    w_evidence: float = 0.0
    w_transfer: float = 0.0
    w_failure_risk: float = 0.0
    w_redundancy: float = 0.0
    w_cost: float = 0.0

    # Final combined utility
    total_utility: float = 0.0
    transfer_assessment: TransferAssessment = TransferAssessment.INSUFFICIENT_EVIDENCE
    explanation: str = ""
    metadata: Dict[str, Any] | None = None

    @classmethod
    def compute(
        cls,
        action_id: str,
        action_type: str,
        raw_scores: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        transfer_assessment: TransferAssessment = TransferAssessment.INSUFFICIENT_EVIDENCE,
        metadata: Optional[Dict[str, Any]] = None,
        raw_components: Optional[Dict[str, float]] = None,
    ) -> "ScoreDecomposition":
        raw = raw_scores or raw_components or {}
        w = weights or {}
        perf = float(raw.get("performance", 0.0))
        info = float(raw.get("information_gain", 0.0))
        nov = float(raw.get("novelty", 0.0))
        ev = float(raw.get("evidence", 0.0))
        trans = float(raw.get("transfer", 0.0))
        risk = float(raw.get("failure_risk", 0.0))
        red = float(raw.get("redundancy", 0.0))
        cost = float(raw.get("cost", 0.0))

        wp = w.get("w_performance", 1.0) * perf
        wi = w.get("w_information_gain", 0.8) * info
        wn = w.get("w_novelty", 0.5) * nov
        we = w.get("w_evidence", 0.7) * ev
        wt = w.get("w_transfer", 0.6) * trans
        wf = w.get("w_failure_risk", 1.2) * risk
        wr = w.get("w_redundancy", 0.8) * red
        wc = w.get("w_cost", 0.3) * cost

        utility = (wp + wi + wn + we + wt) - (wf + wr + wc)

        # Deterministic human-readable explanation
        explanation = (
            f"Action '{action_id}' ({action_type}): utility={utility:.4f} "
            f"[+perf:{wp:.3f}, +info:{wi:.3f}, +nov:{wn:.3f}, +ev:{we:.3f}, +trans:{wt:.3f} | "
            f"-risk:{wf:.3f}, -red:{wr:.3f}, -cost:{wc:.3f}]. "
            f"Transfer: {transfer_assessment.value}."
        )

        return cls(
            id=f"score_{action_id}",
            schema_version="1.0",
            action_id=action_id,
            action_type=action_type,
            performance=perf,
            information_gain=info,
            novelty=nov,
            evidence=ev,
            transfer=trans,
            failure_risk=risk,
            redundancy=red,
            cost=cost,
            w_performance=wp,
            w_information_gain=wi,
            w_novelty=wn,
            w_evidence=we,
            w_transfer=wt,
            w_failure_risk=wf,
            w_redundancy=wr,
            w_cost=wc,
            total_utility=utility,
            transfer_assessment=transfer_assessment,
            explanation=explanation,
            metadata=metadata or {},
        )

    @property
    def raw_components(self) -> Dict[str, float]:
        return {
            "performance": self.performance,
            "information_gain": self.information_gain,
            "novelty": self.novelty,
            "evidence": self.evidence,
            "transfer": self.transfer,
            "failure_risk": self.failure_risk,
            "redundancy": self.redundancy,
            "cost": self.cost,
        }

    @property
    def weighted_contributions(self) -> Dict[str, float]:
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

    def verify_mathematical_consistency(self, tolerance: float = 1e-4) -> bool:
        """Verify that total_utility strictly equals positive minus penalty contributions."""
        expected = (
            self.w_performance
            + self.w_information_gain
            + self.w_novelty
            + self.w_evidence
            + self.w_transfer
        ) - (self.w_failure_risk + self.w_redundancy + self.w_cost)
        return abs(self.total_utility - expected) <= tolerance

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ScoreDecomposition":
        d = dict(obj)
        if "transfer_assessment" in d and isinstance(d["transfer_assessment"], str):
            d["transfer_assessment"] = TransferAssessment(d["transfer_assessment"])
        return cls(**d)
