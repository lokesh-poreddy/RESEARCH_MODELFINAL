"""Canonical Claim domain contract.

A Claim represents a scientific assertion (e.g. "Operator X improves metric Y on task Z").
Evidence represents empirical or literature support, contradiction, or uncertainty regarding that assertion.

Claim ≠ Evidence.

A claim may be supported by some evidence and contradicted by other evidence simultaneously.
Contradictions are explicitly preserved and never averaged away.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import DomainObject


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNCERTAIN = "UNCERTAIN"
    SPECULATIVE = "SPECULATIVE"


class EvidenceRelation(str, Enum):
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTED_BY = "CONTRADICTED_BY"
    UNCERTAIN_FROM = "UNCERTAIN_FROM"
    SPECULATIVE_FROM = "SPECULATIVE_FROM"


@dataclass(frozen=True)
class Claim(DomainObject):
    """A scientific assertion with explicit evidence relationships."""
    statement: str
    claim_type: str | None = None
    confidence: float | None = None
    status: str = ClaimStatus.SPECULATIVE.value
    hypothesis_id: str | None = None
    problem_id: str | None = None
    supporting_evidence_ids: list[str] | None = None
    contradicting_evidence_ids: list[str] | None = None
    uncertain_evidence_ids: list[str] | None = None
    speculative_evidence_ids: list[str] | None = None
    provenance_id: str | None = None
    created_at: float | None = None
    metadata: Dict[str, Any] | None = None

    @property
    def has_contradiction(self) -> bool:
        """True if the claim has both supporting and contradicting evidence."""
        supp = bool(self.supporting_evidence_ids)
        contra = bool(self.contradicting_evidence_ids)
        return supp and contra

    def assess_status(self) -> str:
        """Deterministically assess claim status based on all attached evidence."""
        n_supp = len(self.supporting_evidence_ids or [])
        n_contra = len(self.contradicting_evidence_ids or [])
        n_spec = len(self.speculative_evidence_ids or [])

        if n_supp > 0 and n_contra > 0:
            # Contradiction explicitly recognized — remains UNCERTAIN pending resolution
            return ClaimStatus.UNCERTAIN.value
        if n_supp > 0 and n_contra == 0:
            return ClaimStatus.SUPPORTED.value
        if n_contra > 0 and n_supp == 0:
            return ClaimStatus.CONTRADICTED.value
        if n_spec > 0 and n_supp == 0 and n_contra == 0:
            return ClaimStatus.SPECULATIVE.value
        return ClaimStatus.UNCERTAIN.value

    def add_evidence(
        self,
        evidence_or_id: str | Any,
        relation: str | EvidenceRelation | None = None,
    ) -> "Claim":
        """Return a new frozen Claim with the evidence attached and status reassessed."""
        if hasattr(evidence_or_id, "id"):
            evidence_id = evidence_or_id.id
            if relation is None and hasattr(evidence_or_id, "relation") and evidence_or_id.relation:
                relation = evidence_or_id.relation
        else:
            evidence_id = str(evidence_or_id)

        if relation is None:
            relation = EvidenceRelation.SUPPORTED_BY

        rel_str = relation.value if isinstance(relation, EvidenceRelation) else str(relation)

        supp = list(self.supporting_evidence_ids or [])
        contra = list(self.contradicting_evidence_ids or [])
        unc = list(self.uncertain_evidence_ids or [])
        spec = list(self.speculative_evidence_ids or [])

        if rel_str == EvidenceRelation.SUPPORTED_BY.value:
            if evidence_id not in supp:
                supp.append(evidence_id)
        elif rel_str == EvidenceRelation.CONTRADICTED_BY.value:
            if evidence_id not in contra:
                contra.append(evidence_id)
        elif rel_str == EvidenceRelation.SPECULATIVE_FROM.value:
            if evidence_id not in spec:
                spec.append(evidence_id)
        else:
            if evidence_id not in unc:
                unc.append(evidence_id)

        candidate = Claim(
            id=self.id,
            schema_version=self.schema_version,
            statement=self.statement,
            claim_type=self.claim_type,
            confidence=self.confidence,
            hypothesis_id=self.hypothesis_id,
            problem_id=self.problem_id,
            supporting_evidence_ids=supp or None,
            contradicting_evidence_ids=contra or None,
            uncertain_evidence_ids=unc or None,
            speculative_evidence_ids=spec or None,
            provenance_id=self.provenance_id,
            created_at=self.created_at or time.time(),
            metadata=self.metadata,
        )
        new_status = candidate.assess_status()
        return Claim(
            id=candidate.id,
            schema_version=candidate.schema_version,
            statement=candidate.statement,
            claim_type=candidate.claim_type,
            confidence=candidate.confidence,
            status=new_status,
            hypothesis_id=candidate.hypothesis_id,
            problem_id=candidate.problem_id,
            supporting_evidence_ids=candidate.supporting_evidence_ids,
            contradicting_evidence_ids=candidate.contradicting_evidence_ids,
            uncertain_evidence_ids=candidate.uncertain_evidence_ids,
            speculative_evidence_ids=candidate.speculative_evidence_ids,
            provenance_id=candidate.provenance_id,
            created_at=candidate.created_at,
            metadata=candidate.metadata,
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Claim":
        return cls(**obj)
