"""researchforge/evidence/__init__.py — Evidence package.

RF-1.0.0-alpha.2.1 / RF-1.0.0-alpha.3 (Phase 8):
  - Canonical domain contracts for scientific evidence and claims:
    - Evidence: adjudicated, typed, provenance-linked scientific evidence
    - EvidenceType: categorical taxonomy of evidence
    - EvidenceQuality: multi-dimensional quality assessment
    - Claim: research assertion
    - ClaimStatus: SUPPORTED, CONTRADICTED, UNCERTAIN, SPECULATIVE
    - EvidenceRelation: SUPPORTED_BY, CONTRADICTED_BY, UNCERTAIN_FROM, SPECULATIVE_FROM
  - Pipeline & Normalization:
    - EvidenceNormalizer: normalizes experiments, negative results, trajectories, literature, replications
    - EvidenceStore: deterministic repository with deduplication and contradiction preservation
    - EvidenceSnapshot: temporal snapshot preventing retroactive leakage
  - Candidates:
    - EvidenceCandidate: unadjudicated search hit
"""
from .evidence import (
    Evidence,
    EVIDENCE_SCHEMA,
    EvidenceCandidate,
    EVIDENCE_CANDIDATE_SCHEMA,
)
from .normalizer import EvidenceNormalizer
from .store import EvidenceStore
from .snapshot import EvidenceSnapshot
from ..domain.evidence import (
    Evidence as CanonicalEvidence,
    EvidenceType,
    EvidenceQuality,
)
from ..domain.claim import (
    Claim,
    ClaimStatus,
    EvidenceRelation,
)

__all__ = [
    "Evidence",
    "CanonicalEvidence",
    "EvidenceType",
    "EvidenceQuality",
    "Claim",
    "ClaimStatus",
    "EvidenceRelation",
    "EVIDENCE_SCHEMA",
    "EvidenceCandidate",
    "EVIDENCE_CANDIDATE_SCHEMA",
    "EvidenceNormalizer",
    "EvidenceStore",
    "EvidenceSnapshot",
]
