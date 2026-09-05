"""researchforge/transfer/types.py — Domain types, knowledge categories, and transfer decision records.

Phase 13 (RF-1.0.0-beta.1):
Enforces typed knowledge categorisation, separating portable algorithmic/procedural
invariants from task-specific parameterisations and assumptions.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class TransferableKnowledgeCategory(str, Enum):
    """Categorisation of knowledge elements stored in or retrieved from ECRM/RDG."""
    # Portable categories allowed in SELECTIVE mode
    RESEARCH_PROCEDURE = "RESEARCH_PROCEDURE"
    EVALUATION_PROCEDURE = "EVALUATION_PROCEDURE"
    GENERIC_FAILURE_PATTERN = "GENERIC_FAILURE_PATTERN"
    REPRODUCIBILITY_LESSON = "REPRODUCIBILITY_LESSON"
    INVARIANT_ALGORITHMIC_PRINCIPLE = "INVARIANT_ALGORITHMIC_PRINCIPLE"

    # Task-specific categories restricted in SELECTIVE mode across family boundaries
    TASK_SPECIFIC_HYPOTHESIS = "TASK_SPECIFIC_HYPOTHESIS"
    FEATURE_ASSUMPTIONS = "FEATURE_ASSUMPTIONS"
    ARCHITECTURE_ASSUMPTIONS = "ARCHITECTURE_ASSUMPTIONS"
    TASK_SPECIFIC_HYPERPARAMETERS = "TASK_SPECIFIC_HYPERPARAMETERS"
    TASK_SPECIFIC_LABELS = "TASK_SPECIFIC_LABELS"


ALLOWED_SELECTIVE_CATEGORIES: Set[TransferableKnowledgeCategory] = frozenset([
    TransferableKnowledgeCategory.RESEARCH_PROCEDURE,
    TransferableKnowledgeCategory.EVALUATION_PROCEDURE,
    TransferableKnowledgeCategory.GENERIC_FAILURE_PATTERN,
    TransferableKnowledgeCategory.REPRODUCIBILITY_LESSON,
    TransferableKnowledgeCategory.INVARIANT_ALGORITHMIC_PRINCIPLE,
])

RESTRICTED_SELECTIVE_CATEGORIES: Set[TransferableKnowledgeCategory] = frozenset([
    TransferableKnowledgeCategory.TASK_SPECIFIC_HYPOTHESIS,
    TransferableKnowledgeCategory.FEATURE_ASSUMPTIONS,
    TransferableKnowledgeCategory.ARCHITECTURE_ASSUMPTIONS,
    TransferableKnowledgeCategory.TASK_SPECIFIC_HYPERPARAMETERS,
    TransferableKnowledgeCategory.TASK_SPECIFIC_LABELS,
])


class TransferGuardMode(str, Enum):
    """Operational mode for TransferGuard."""
    STRICT = "STRICT"        # Suppresses all prior task memories (clean cold-start isolation)
    SELECTIVE = "SELECTIVE"  # Retains only portable knowledge categories; filters task-specific assumptions
    DISABLED = "DISABLED"    # Unguarded transfer (preserves naive ECRM behavior)


@dataclass(frozen=True)
class TransferDecisionRecord:
    """Immutable audit record detailing why memory was permitted, restricted, or suppressed."""
    source_task_id: str
    target_task_id: str
    source_family: str
    target_family: str
    prior_affinity: float
    historical_transfer_effect: Optional[float]
    evidence_volume: int
    transfer_score: float
    guard_mode: TransferGuardMode
    action_taken: str  # "ALLOW_ALL" | "SUPPRESS_ALL" | "FILTER_TYPED" | "ALLOW_EMPIRICAL_EXCEPTION"
    allowed_memory_ids: List[str]
    filtered_memory_ids: List[str]
    rationale: str
    timestamp: float = field(default_factory=time.time)
    provenance_id: str = ""

    def __post_init__(self) -> None:
        if not self.provenance_id:
            payload = {
                "source_task_id": self.source_task_id,
                "target_task_id": self.target_task_id,
                "prior_affinity": round(self.prior_affinity, 4),
                "historical_transfer_effect": round(self.historical_transfer_effect, 4) if self.historical_transfer_effect is not None else None,
                "evidence_volume": self.evidence_volume,
                "transfer_score": round(self.transfer_score, 4),
                "guard_mode": self.guard_mode.value,
                "action_taken": self.action_taken,
                "allowed_count": len(self.allowed_memory_ids),
                "filtered_count": len(self.filtered_memory_ids),
            }
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            object.__setattr__(self, "provenance_id", hashlib.sha256(raw).hexdigest()[:16])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_task_id": self.source_task_id,
            "target_task_id": self.target_task_id,
            "source_family": self.source_family,
            "target_family": self.target_family,
            "prior_affinity": self.prior_affinity,
            "historical_transfer_effect": self.historical_transfer_effect,
            "evidence_volume": self.evidence_volume,
            "transfer_score": self.transfer_score,
            "guard_mode": self.guard_mode.value,
            "action_taken": self.action_taken,
            "allowed_memory_ids": list(self.allowed_memory_ids),
            "filtered_memory_ids": list(self.filtered_memory_ids),
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "provenance_id": self.provenance_id,
        }
