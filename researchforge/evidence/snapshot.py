"""Decision-time evidence snapshot contract.

Scientific Objective (Phase 8):
Provide an immutable, provenance-linked snapshot of what evidence was
available when a specific research decision was made. Ensures strict
temporal isolation and prevents retroactive information leakage into
historical decisions.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..domain.base import DomainObject
from ..domain.evidence import Evidence


@dataclass(frozen=True)
class EvidenceSnapshot(DomainObject):
    """Immutable, content-addressed snapshot of evidence available at decision time."""
    decision_id: str
    as_of_timestamp: float
    evidence_ids: tuple[str, ...]
    provenance_id: str | None = None
    metadata: Dict[str, Any] | None = None
    evidence_items: tuple[Evidence, ...] | None = None

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_ids)

    def includes_evidence(self, evidence_id: str) -> bool:
        return evidence_id in self.evidence_ids

    @classmethod
    def create(
        cls,
        decision_id: str,
        as_of_timestamp: float,
        evidence_ids: List[str],
        provenance_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_items: Optional[List[Evidence]] = None,
    ) -> "EvidenceSnapshot":
        sorted_ids = tuple(sorted(set(evidence_ids)))
        content = f"{decision_id}:{as_of_timestamp}:{','.join(sorted_ids)}"
        snap_id = f"snap_{hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]}"
        return cls(
            id=snap_id,
            schema_version="1",
            decision_id=decision_id,
            as_of_timestamp=as_of_timestamp,
            evidence_ids=sorted_ids,
            provenance_id=provenance_id,
            metadata=metadata,
            evidence_items=tuple(evidence_items) if evidence_items else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["evidence_ids"] = list(self.evidence_ids)
        if self.evidence_items is not None:
            base["evidence_items"] = [e.to_dict() for e in self.evidence_items]
        return base

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "EvidenceSnapshot":
        data = dict(obj)
        if "evidence_ids" in data and isinstance(data["evidence_ids"], list):
            data["evidence_ids"] = tuple(data["evidence_ids"])
        if "evidence_items" in data and isinstance(data["evidence_items"], list):
            data["evidence_items"] = tuple(
                Evidence.from_dict(item) if isinstance(item, dict) else item
                for item in data["evidence_items"]
            )
        return cls(**data)
