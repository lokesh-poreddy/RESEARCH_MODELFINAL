from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any
from researchforge.domain.base import DomainObject


class RelationType(str, Enum):
    DERIVED_FROM = "DERIVED_FROM"
    ADDRESSES = "ADDRESSES"
    GENERATED = "GENERATED"
    MOTIVATED_BY = "MOTIVATED_BY"
    SELECTED_BY = "SELECTED_BY"
    IMPLEMENTS = "IMPLEMENTS"
    EVALUATES = "EVALUATES"
    EXECUTED_AS = "EXECUTED_AS"
    PRODUCED = "PRODUCED"
    DIAGNOSED_BY = "DIAGNOSED_BY"
    FAILED_AS = "FAILED_AS"
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTED_BY = "CONTRADICTED_BY"
    UNCERTAIN_FROM = "UNCERTAIN_FROM"
    SPECULATIVE_FROM = "SPECULATIVE_FROM"
    VALIDATED_BY = "VALIDATED_BY"
    INVALIDATED_BY = "INVALIDATED_BY"
    NEXT_STATE = "NEXT_STATE"
    PRECEDES = "PRECEDES"
    BRANCH_OF = "BRANCH_OF"
    RECOVERED_FROM = "RECOVERED_FROM"
    REPLICATES = "REPLICATES"
    TRANSFERRED_FROM = "TRANSFERRED_FROM"
    USES_GENOME = "USES_GENOME"
    USES_DATASET = "USES_DATASET"
    HAS_PROVENANCE = "HAS_PROVENANCE"


@dataclass(frozen=True)
class GraphEdge(DomainObject):
    source_id: str
    target_id: str
    relation: str
    provenance_id: str | None = None
    confidence: float | None = None
    metadata: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "GraphEdge":
        return cls(**obj)
