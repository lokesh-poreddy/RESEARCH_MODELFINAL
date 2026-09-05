"""researchforge/transfer/ — TransferGuard and selective experience gating.

Phase 13 (RF-1.0.0-beta.1):
Continuous adaptation, transfer confidence gating, and negative transfer suppression.
"""
from .types import (
    TransferableKnowledgeCategory,
    ALLOWED_SELECTIVE_CATEGORIES,
    RESTRICTED_SELECTIVE_CATEGORIES,
    TransferGuardMode,
    TransferDecisionRecord,
)
from .affinity import (
    TaskProfile,
    compute_prior_affinity,
    compute_historical_transfer_effect,
    compute_transfer_score,
)
from .guard import TransferGuard
from .counterfactual import (
    CounterfactualStatus,
    CounterfactualEvaluation,
    CounterfactualArbitrator,
)

__all__ = [
    "TransferableKnowledgeCategory",
    "ALLOWED_SELECTIVE_CATEGORIES",
    "RESTRICTED_SELECTIVE_CATEGORIES",
    "TransferGuardMode",
    "TransferDecisionRecord",
    "TaskProfile",
    "compute_prior_affinity",
    "compute_historical_transfer_effect",
    "compute_transfer_score",
    "TransferGuard",
    "CounterfactualStatus",
    "CounterfactualEvaluation",
    "CounterfactualArbitrator",
]
