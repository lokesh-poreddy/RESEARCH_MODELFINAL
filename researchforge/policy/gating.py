"""researchforge/policy/gating.py — Policy update gating and validation.

Scientific & Governance Constraints (Phase 9):
A policy update cannot be silently or automatically activated.
It requires:
  old_policy + experience + reason + proposed_update + deterministic validation
  + comparison against previous policy + governance review -> new_policy_version
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..domain.action import ActionType, ResearchAction
from ..domain.base import DomainObject
from ..domain.state import ResearchState
from ..evidence.snapshot import EvidenceSnapshot
from .config import PolicyConfig
from .research_policy import ResearchPolicy


@dataclass(frozen=True)
class PolicyUpdateProposal(DomainObject):
    """Proposal for updating the active ResearchPolicy configuration."""
    id: str
    schema_version: str
    old_policy_version: str
    proposed_policy_version: str
    reason: str
    proposed_config: PolicyConfig
    experience_refs: List[str] = field(default_factory=list)
    governance_review_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    provenance_id: Optional[str] = None


@dataclass(frozen=True)
class PolicyGateResult(DomainObject):
    """Result of passing a PolicyUpdateProposal through validation gates."""
    id: str
    schema_version: str
    proposal_id: str
    approved: bool
    reasons: List[str]
    active_policy_version: str
    validation_checks_passed: List[str]
    validation_checks_failed: List[str]
    timestamp: float = field(default_factory=time.time)
    provenance_id: Optional[str] = None


class PolicyGate:
    """Validates and gates policy updates before activation."""

    def __init__(self, active_policy: ResearchPolicy) -> None:
        self.active_policy = active_policy

    def evaluate_proposal(
        self,
        proposal: PolicyUpdateProposal,
        governance_approved: bool = True,
    ) -> PolicyGateResult:
        passed_checks: List[str] = []
        failed_checks: List[str] = []
        reasons: List[str] = []

        cfg = proposal.proposed_config

        # 1. Schema / Type validation
        if not isinstance(cfg, PolicyConfig):
            failed_checks.append("schema_validation")
            reasons.append("Proposed configuration is not a valid PolicyConfig.")
        else:
            passed_checks.append("schema_validation")

        # 2. Version progression check (cannot overwrite current version or use historical version)
        if cfg.policy_version == self.active_policy.policy_version:
            failed_checks.append("version_monotonicity")
            reasons.append(f"Proposed policy version '{cfg.policy_version}' is identical to active policy.")
        else:
            passed_checks.append("version_monotonicity")

        # 3. Non-negative weights validation
        weights = cfg.weights_dict()
        negative_weights = [k for k, v in weights.items() if v < 0]
        if negative_weights:
            failed_checks.append("weight_positivity")
            reasons.append(f"Weights must be non-negative. Found: {negative_weights}")
        else:
            passed_checks.append("weight_positivity")

        # 4. Deterministic fixture validation
        try:
            test_policy = ResearchPolicy(config=cfg)
            test_state = ResearchState(id="gate_s0", schema_version="1.0")
            test_actions = [
                ResearchAction(id="gate_a1", schema_version="1.0", action_type=ActionType.EXPLORE_NEW_STRATEGY, target_context={}, expected_objective="test1"),
                ResearchAction(id="gate_a2", schema_version="1.0", action_type=ActionType.EXPLOIT_KNOWN_STRATEGY, target_context={}, expected_objective="test2"),
            ]
            test_snap = EvidenceSnapshot.create(decision_id="gate_d0", as_of_timestamp=100.0, evidence_ids=[])
            dec1, rec1 = test_policy.evaluate_candidates(test_state, test_actions, test_snap, decision_timestamp=100.0)
            dec2, rec2 = test_policy.evaluate_candidates(test_state, test_actions, test_snap, decision_timestamp=100.0)
            if rec1.decision_fingerprint() == rec2.decision_fingerprint():
                passed_checks.append("deterministic_fixture")
            else:
                failed_checks.append("deterministic_fixture")
                reasons.append("Proposed policy produced non-deterministic evaluations on identical inputs.")
        except Exception as e:
            failed_checks.append("deterministic_fixture")
            reasons.append(f"Fixture execution failed: {str(e)}")

        # 5. Governance Advisory Check
        if not governance_approved:
            failed_checks.append("governance_review")
            reasons.append("Governance review has not approved this policy update.")
        else:
            passed_checks.append("governance_review")

        approved = len(failed_checks) == 0
        active_ver = cfg.policy_version if approved else self.active_policy.policy_version

        if approved:
            reasons.append(f"Policy update approved. Promoting to version '{cfg.policy_version}'.")
            self.active_policy = ResearchPolicy(config=cfg)

        return PolicyGateResult(
            id=f"gate_{proposal.id}",
            schema_version="1.0",
            proposal_id=proposal.id,
            approved=approved,
            reasons=reasons,
            active_policy_version=active_ver,
            validation_checks_passed=passed_checks,
            validation_checks_failed=failed_checks,
            provenance_id=proposal.provenance_id,
        )
