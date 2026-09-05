"""researchforge/governance/contracts.py — Typed contracts for ResearchForge Meta-Development Loop.

Scientific & Architectural Constraints (Phase 9):
The Governance Layer governs changes to ResearchForge itself (Meta-Development Loop),
completely decoupled from the Research Loop which governs actions on target research problems.
Governance is advisory before freeze: it may approve/reject/request changes, but does not
silently or automatically mutate production code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..domain.base import DomainObject


class GovernanceStage(str, Enum):
    """The 9 canonical stages of the ResearchForge Meta-Development Loop."""
    PLAN = "PLAN"
    CHALLENGE = "CHALLENGE"
    IMPLEMENT = "IMPLEMENT"
    REVIEW = "REVIEW"
    TEST = "TEST"
    SCIENTIFIC_VALIDATE = "SCIENTIFIC_VALIDATE"
    BENCHMARK = "BENCHMARK"
    FREEZE = "FREEZE"
    RETROSPECT = "RETROSPECT"


class ReviewerRole(str, Enum):
    """Independent review responsibilities in governance."""
    RESEARCH_ARCHITECT = "RESEARCH_ARCHITECT"
    SCIENTIFIC_METHOD_REVIEWER = "SCIENTIFIC_METHOD_REVIEWER"
    IMPLEMENTATION_ENGINEER = "IMPLEMENTATION_ENGINEER"
    SAFETY_EXECUTION_REVIEWER = "SAFETY_EXECUTION_REVIEWER"
    RESEARCH_VALIDITY_REVIEWER = "RESEARCH_VALIDITY_REVIEWER"
    ADVERSARIAL_EVALUATOR = "ADVERSARIAL_EVALUATOR"
    BENCHMARK_SCIENTIST = "BENCHMARK_SCIENTIST"
    RELEASE_REPRODUCIBILITY_ENGINEER = "RELEASE_REPRODUCIBILITY_ENGINEER"
    RESEARCH_RETROSPECTIVE = "RESEARCH_RETROSPECTIVE"


class ReviewDecision(str, Enum):
    """Controlled decision verdict from a governance review."""
    APPROVE = "APPROVE"
    APPROVE_WITH_WARNINGS = "APPROVE_WITH_WARNINGS"
    REQUEST_CHANGES = "REQUEST_CHANGES"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


class FindingSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class VersionTarget(DomainObject):
    """Exact, immutable target for a governance review.
    
    'latest' or empty strings are strictly disallowed as implicit targets.
    """
    code_revision: str
    rf_version: str
    schema_version: str
    affected_components: tuple[str, ...]
    configuration_fingerprint: str
    benchmark_artifact_fingerprint: str

    def __post_init__(self) -> None:
        if not self.code_revision or self.code_revision.lower() == "latest":
            raise ValueError("VersionTarget requires an explicit, immutable code revision (not 'latest').")
        if not self.rf_version:
            raise ValueError("VersionTarget requires an explicit rf_version.")
        if not self.configuration_fingerprint:
            raise ValueError("VersionTarget requires an explicit configuration_fingerprint.")

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "VersionTarget":
        d = dict(obj)
        if "affected_components" in d and isinstance(d["affected_components"], (list, tuple)):
            d["affected_components"] = tuple(d["affected_components"])
        return cls(**d)


@dataclass(frozen=True)
class ReviewFinding(DomainObject):
    """A structured finding emitted by a governance reviewer."""
    finding_id: str
    category: str
    severity: FindingSeverity
    description: str
    recommended_action: str
    evidence_refs: List[str] = field(default_factory=list)
    status: str = "OPEN"  # OPEN, RESOLVED, ACCEPTED_RISK

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ReviewFinding":
        d = dict(obj)
        if "severity" in d and isinstance(d["severity"], str):
            d["severity"] = FindingSeverity(d["severity"])
        return cls(**d)


@dataclass(frozen=True)
class GovernanceReview(DomainObject):
    """A durable, provenance-bearing review record for a specific stage and role."""
    target: VersionTarget
    reviewer_role: ReviewerRole
    review_stage: GovernanceStage
    findings: List[ReviewFinding]
    decision: ReviewDecision
    severity: FindingSeverity
    rationale: str
    evidence_refs: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    provenance_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "GovernanceReview":
        d = dict(obj)
        if "target" in d and isinstance(d["target"], dict):
            d["target"] = VersionTarget.from_dict(d["target"])
        if "reviewer_role" in d and isinstance(d["reviewer_role"], str):
            d["reviewer_role"] = ReviewerRole(d["reviewer_role"])
        if "review_stage" in d and isinstance(d["review_stage"], str):
            d["review_stage"] = GovernanceStage(d["review_stage"])
        if "decision" in d and isinstance(d["decision"], str):
            d["decision"] = ReviewDecision(d["decision"])
        if "severity" in d and isinstance(d["severity"], str):
            d["severity"] = FindingSeverity(d["severity"])
        if "findings" in d and isinstance(d["findings"], list):
            d["findings"] = [ReviewFinding.from_dict(f) if isinstance(f, dict) else f for f in d["findings"]]
        return cls(**d)


@dataclass(frozen=True)
class ReleaseGate(DomainObject):
    """Final release-gate record evaluated before freezing a version."""
    target: VersionTarget
    passed: bool
    blocking_reasons: List[str]
    completed_stages: List[GovernanceStage]
    reviews: List[GovernanceReview]
    verified_at: float = field(default_factory=time.time)
    provenance_id: Optional[str] = None


@dataclass(frozen=True)
class RetrospectiveRecord(DomainObject):
    """Durable retrospective record capturing learnings from each major phase."""
    phase: str
    target: VersionTarget
    what_changed: List[str]
    what_passed: List[str]
    what_failed: List[str]
    unexpected_behavior: List[str]
    negative_results: List[str]
    architectural_debt: List[str]
    research_insight: str
    benchmark_insight: str
    next_upgrade_hypothesis: str
    created_at: float = field(default_factory=time.time)
    provenance_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
