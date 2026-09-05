"""researchforge/policy/decision_record.py — Auditable policy evaluation record.

Architectural Requirement (Phase 9):
Distinguishes the domain-level ResearchDecision from the detailed PolicyDecisionRecord.
Preserves complete auditable scoring telemetry explaining how candidate actions were ranked,
why an action was selected, and exactly what alternatives were rejected.
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import json

from ..domain.action import ResearchAction
from ..domain.base import DomainObject, _canonical_json
from .score_decomposition import ScoreDecomposition


@dataclass(frozen=True)
class PolicyDecisionRecord(DomainObject):
    """First-class auditable record of a ResearchPolicy evaluation cycle.
    
    Phase 9A Invariant:
    All 18+ telemetry properties are schema-addressable and auditable.
    Fields without valid observations are explicitly marked unavailable / None / not applicable,
    never fabricated.
    """
    id: str
    schema_version: str
    decision_id: str
    research_state_fingerprint: str
    evidence_snapshot_id: str
    policy_version: str
    policy_fingerprint: str
    selected_action_id: str
    selected_action: ResearchAction
    rejected_action_ids: List[str]
    score_decompositions: Dict[str, ScoreDecomposition]
    policy_weights: Dict[str, float]
    decision_timestamp: float
    explanation: str
    available_memory_ref: Optional[str] = None
    provenance_id: Optional[str] = None
    credit_assignment_meta: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Normalized Phase 9A Telemetry Fields
    evidence_snapshot_fingerprint: Optional[str] = None
    memory_retrieval_summary: Optional[Dict[str, Any]] = None
    candidate_actions_list: Optional[List[ResearchAction]] = None
    portfolio_state: Optional[Dict[str, Any]] = None
    baseline_score: Optional[float] = None

    @property
    def candidate_actions(self) -> List[ResearchAction]:
        """All candidate actions considered during this decision cycle."""
        if self.candidate_actions_list:
            return list(self.candidate_actions_list)
        return [self.selected_action]

    @property
    def candidate_component_scores(self) -> Dict[str, Dict[str, float]]:
        """Raw component scores for all candidates."""
        return {k: v.raw_components for k, v in self.score_decompositions.items()}

    @property
    def weighted_score_contributions(self) -> Dict[str, Dict[str, float]]:
        """Weighted score contributions for all candidates."""
        return {k: v.weighted_contributions for k, v in self.score_decompositions.items()}

    @property
    def rejected_alternatives(self) -> List[str]:
        """IDs of rejected alternative actions."""
        return list(self.rejected_action_ids)

    @property
    def predicted_utility(self) -> float:
        """The predicted total utility of the selected action."""
        if self.selected_action_id in self.score_decompositions:
            return self.score_decompositions[self.selected_action_id].total_utility
        return 0.0

    @property
    def memory_contribution(self) -> Optional[float]:
        """Contribution of memory/transfer toward the selected action score."""
        if self.selected_action_id in self.score_decompositions:
            return self.score_decompositions[self.selected_action_id].w_transfer
        return None

    @property
    def evidence_contribution(self) -> Optional[float]:
        """Contribution of empirical evidence toward the selected action score."""
        if self.selected_action_id in self.score_decompositions:
            return self.score_decompositions[self.selected_action_id].w_evidence
        return None

    @property
    def failure_risk_contribution(self) -> Optional[float]:
        """Penalty from failure risk toward the selected action score."""
        if self.selected_action_id in self.score_decompositions:
            return self.score_decompositions[self.selected_action_id].w_failure_risk
        return None

    @property
    def expected_cost(self) -> Optional[float]:
        """Cost associated with the selected action."""
        if self.selected_action_id in self.score_decompositions:
            return self.score_decompositions[self.selected_action_id].cost
        return None

    @property
    def decision_datetime(self) -> datetime:
        """Timezone-aware UTC datetime of the decision."""
        return datetime.fromtimestamp(self.decision_timestamp, tz=timezone.utc)

    @property
    def decision_timestamp_iso(self) -> str:
        """ISO-8601 UTC string representation of decision timestamp."""
        return self.decision_datetime.isoformat()

    def decision_fingerprint(self) -> str:
        """Deterministic fingerprint of this policy evaluation."""
        payload = {
            "schema_version": self.schema_version,
            "research_state_fingerprint": self.research_state_fingerprint,
            "evidence_snapshot_id": self.evidence_snapshot_id,
            "policy_version": self.policy_version,
            "policy_fingerprint": self.policy_fingerprint,
            "selected_action_id": self.selected_action_id,
            "rejected_action_ids": sorted(self.rejected_action_ids),
            "scores": {k: v.total_utility for k, v in sorted(self.score_decompositions.items())},
        }
        j = _canonical_json(payload)
        return hashlib.sha256(j.encode("utf-8")).hexdigest()

    def telemetry_dict(self) -> Dict[str, Any]:
        """Schema-addressable export of all 20 required research telemetry fields."""
        return {
            "research_state_fingerprint": self.research_state_fingerprint,
            "evidence_snapshot_id": self.evidence_snapshot_id,
            "evidence_snapshot_fingerprint": self.evidence_snapshot_fingerprint,
            "memory_retrieval_summary": self.memory_retrieval_summary,
            "candidate_actions": [a.to_dict() for a in self.candidate_actions],
            "candidate_component_scores": self.candidate_component_scores,
            "weighted_score_contributions": self.weighted_score_contributions,
            "policy_version": self.policy_version,
            "policy_fingerprint": self.policy_fingerprint,
            "portfolio_state": self.portfolio_state,
            "selected_action": self.selected_action.to_dict(),
            "selected_action_id": self.selected_action_id,
            "rejected_alternatives": self.rejected_alternatives,
            "predicted_utility": self.predicted_utility,
            "baseline_score": self.baseline_score,
            "memory_contribution": self.memory_contribution,
            "evidence_contribution": self.evidence_contribution,
            "failure_risk_contribution": self.failure_risk_contribution,
            "expected_cost": self.expected_cost,
            "provenance_id": self.provenance_id,
            "decision_timestamp": self.decision_timestamp_iso,
        }

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PolicyDecisionRecord":
        d = dict(obj)
        if "selected_action" in d and isinstance(d["selected_action"], dict):
            d["selected_action"] = ResearchAction.from_dict(d["selected_action"])
        if "candidate_actions_list" in d and isinstance(d["candidate_actions_list"], list):
            d["candidate_actions_list"] = [
                ResearchAction.from_dict(a) if isinstance(a, dict) else a
                for a in d["candidate_actions_list"]
            ]
        if "score_decompositions" in d and isinstance(d["score_decompositions"], dict):
            d["score_decompositions"] = {
                k: ScoreDecomposition.from_dict(v) if isinstance(v, dict) else v
                for k, v in d["score_decompositions"].items()
            }
        return cls(**d)
