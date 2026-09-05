"""Canonical domain contract import surface for ResearchForge.

This module exposes the canonical, versioned, serializable domain objects
used by the RF-1.0.0-alpha.3 implementation. These are contracts only — no
business logic belongs here.
"""
from .base import DomainObject
from .primitives import EntityId, Version, Timestamp, Confidence, Metric, ArtifactRef
from .validity import ValidityVerdict, Validity
from .provenance import Provenance
from .problem import ResearchProblem
from .question import ResearchQuestion
from .hypothesis import Hypothesis
from .decision import Decision
from .genome import TargetModelGenome, ResearchSystemGenome
from .experiment import ExperimentSpec, ExperimentRun
from .outcome import Outcome
from .diagnosis import Diagnosis
from .failure import Failure
from .evidence import Evidence, EvidenceType, EvidenceQuality
from .claim import Claim, ClaimStatus, EvidenceRelation
from .state import ResearchState
from .action import ResearchAction, ActionType

__all__ = [
    "DomainObject",
    "EntityId",
    "Version",
    "Timestamp",
    "Confidence",
    "Metric",
    "ArtifactRef",
    "ValidityVerdict",
    "Validity",
    "Provenance",
    "ResearchProblem",
    "ResearchQuestion",
    "Hypothesis",
    "Decision",
    "TargetModelGenome",
    "ResearchSystemGenome",
    "ExperimentSpec",
    "ExperimentRun",
    "Outcome",
    "Diagnosis",
    "Failure",
    "Evidence",
    "EvidenceType",
    "EvidenceQuality",
    "Claim",
    "ClaimStatus",
    "EvidenceRelation",
    "ResearchState",
    "ResearchAction",
    "ActionType",
]

