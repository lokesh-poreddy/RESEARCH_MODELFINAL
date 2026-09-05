"""Canonical Evidence domain contract.

Evidence represents an empirical observation, literature finding, experiment outcome,
negative result, or trajectory summary that provides support, contradiction, or uncertainty
for a scientific claim or research decision.

Scientific Rules (Phase 8):
1. Context Specificity ≠ Evidence Sufficiency ≠ Statistical Confidence.
2. High retrieval score ≠ High scientific confidence.
3. Quality is multi-dimensional (source reliability, relevance, empirical support,
   replication status, freshness, contradiction status, provenance completeness);
   a composite score is provided only with all components remaining inspectable.
4. Execution failures and invalid experiments do not produce positive support.
5. Negative evidence is a first-class citizen and explicitly retrievable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .base import DomainObject


class EvidenceType(str, Enum):
    LITERATURE = "literature"
    EXPERIMENTAL = "experimental"
    NEGATIVE = "negative"
    REPLICATION = "replication"
    TRAJECTORY = "trajectory"
    COMPUTATIONAL = "computational"
    DERIVED = "derived"


@dataclass(frozen=True)
class EvidenceQuality:
    """Multi-dimensional evidence quality and reliability assessment."""
    source_reliability: float = 1.0       # Reliability of source in [0, 1]
    relevance: float = 1.0                # Relevance to target claim in [0, 1]
    empirical_support: float = 1.0        # Statistical / metric strength in [0, 1]
    replication_status: str = "untested"  # "untested" | "replicated" | "failed_replication"
    freshness: float = 1.0                # Temporal recency / decay in [0, 1]
    contradiction_status: str = "none"    # "none" | "has_contradiction" | "reconciled"
    provenance_completeness: float = 1.0  # Fraction of provenance chain intact in [0, 1]

    def composite_score(self) -> float:
        """Calculate composite quality where every component remains inspectable."""
        rep_factor = (
            1.0 if self.replication_status == "replicated"
            else (0.4 if self.replication_status == "failed_replication" else 0.8)
        )
        contra_factor = 0.5 if self.contradiction_status == "has_contradiction" else 1.0
        score = (
            0.25 * self.source_reliability +
            0.25 * self.relevance +
            0.25 * self.empirical_support +
            0.15 * self.provenance_completeness +
            0.10 * self.freshness
        ) * rep_factor * contra_factor
        return round(max(0.0, min(1.0, score)), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_reliability": self.source_reliability,
            "relevance": self.relevance,
            "empirical_support": self.empirical_support,
            "replication_status": self.replication_status,
            "freshness": self.freshness,
            "contradiction_status": self.contradiction_status,
            "provenance_completeness": self.provenance_completeness,
            "composite_score": self.composite_score(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceQuality":
        return cls(
            source_reliability=float(d.get("source_reliability", 1.0)),
            relevance=float(d.get("relevance", 1.0)),
            empirical_support=float(d.get("empirical_support", 1.0)),
            replication_status=str(d.get("replication_status", "untested")),
            freshness=float(d.get("freshness", 1.0)),
            contradiction_status=str(d.get("contradiction_status", "none")),
            provenance_completeness=float(d.get("provenance_completeness", 1.0)),
        )


@dataclass(frozen=True)
class Evidence(DomainObject):
    """Canonical, versioned, provenance-linked Evidence contract."""
    source: str
    source_id: str | None = None
    retrieval_timestamp: str | None = None
    retriever: str | None = None
    source_url: str | None = None
    text_location: str | None = None
    quality_indicators: Dict[str, Any] | None = None
    claim_relationship: str | None = None
    confidence: float | None = None
    snippet: str | None = None
    metadata: Dict[str, Any] | None = None

    # Phase 8 Extensions
    evidence_type: str | None = None           # One of EvidenceType
    source_type: str | None = None             # "literature" | "experiment" | "run" | "outcome" | "failure" | "trajectory"
    claim_id: str | None = None                # Target claim ID or reference
    relation: str | None = None                # "SUPPORTED_BY" | "CONTRADICTED_BY" | "UNCERTAIN_FROM" | "SPECULATIVE_FROM"
    provenance_id: str | None = None           # Explicit Provenance link
    quality: EvidenceQuality | None = None     # Structured multi-dimensional quality
    originating_vrdeg_node_ids: list[str] | None = None
    originating_experiment_ids: list[str] | None = None
    originating_run_ids: list[str] | None = None
    originating_outcome_ids: list[str] | None = None
    originating_trajectory_ids: list[str] | None = None
    created_at: float | None = None

    @property
    def evidence_id(self) -> str:
        """Alias for domain id."""
        return self.id

    def semantic_key(self) -> Tuple[Any, ...]:
        """Deterministic tuple for deduplication without collapsing distinct observations."""
        return (
            self.source,
            self.source_id,
            self.evidence_type or "",
            self.claim_id or "",
            self.relation or self.claim_relationship or "",
            self.source_url or "",
            tuple(sorted(self.originating_run_ids or [])),
            tuple(sorted(self.originating_trajectory_ids or [])),
            tuple(sorted(self.originating_experiment_ids or [])),
            round(self.quality.composite_score(), 4) if self.quality else (round(self.confidence, 4) if self.confidence else 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        if self.quality is not None:
            base["quality"] = self.quality.to_dict()
            if base.get("quality_indicators") is None:
                base["quality_indicators"] = self.quality.to_dict()
            if base.get("confidence") is None:
                base["confidence"] = self.quality.composite_score()
        if self.relation is not None and base.get("claim_relationship") is None:
            base["claim_relationship"] = self.relation
        return base

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Evidence":
        data = dict(obj)
        if "id" not in data and "evidence_id" in data:
            data["id"] = data.pop("evidence_id")
        else:
            data.pop("evidence_id", None)
        if "quality" in data and data["quality"] is not None and isinstance(data["quality"], dict):
            data["quality"] = EvidenceQuality.from_dict(data["quality"])
        return cls(**data)

