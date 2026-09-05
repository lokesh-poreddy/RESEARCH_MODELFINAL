"""researchforge/policy/portfolio.py — Research Portfolio with multi-branch diversity.

Scientific Objective (Phase 9):
Maintains multiple active and competing research branches instead of greedily collapsing
to a single branch. Preserves negative/failed branches for historical learning.
Calculates meaningful diversity across strategy families, model families, and intervention types.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..domain.action import ResearchAction
from ..domain.base import DomainObject


class BranchState(str, Enum):
    ACTIVE = "ACTIVE"
    PRUNED = "PRUNED"
    FAILED = "FAILED"
    SATURATED = "SATURATED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class PortfolioBranch(DomainObject):
    """A persistent research branch in the portfolio."""
    originating_hypothesis_id: str
    action: ResearchAction
    score: float
    expected_value: float
    resource_allocation: float  # Fraction of budget [0.0, 1.0]
    diversity_features: Dict[str, str]  # e.g., {"strategy_family": ..., "model_family": ..., "representation": ...}
    branch_state: BranchState = BranchState.ACTIVE
    history_experiment_ids: List[str] = field(default_factory=list)
    failure_history: List[str] = field(default_factory=list)
    provenance_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PortfolioBranch":
        d = dict(obj)
        if "action" in d and isinstance(d["action"], dict):
            d["action"] = ResearchAction.from_dict(d["action"])
        if "branch_state" in d and isinstance(d["branch_state"], str):
            d["branch_state"] = BranchState(d["branch_state"])
        return cls(**d)


class ResearchPortfolio:
    """Manages multi-branch research exploration with negative branch retention and diversity tracking."""

    def __init__(self, max_active_branches: int = 5) -> None:
        self.max_active_branches = max_active_branches
        self._branches: Dict[str, PortfolioBranch] = {}

    def add_branch(self, branch: PortfolioBranch) -> None:
        """Add or update a branch. Negative/failed branches are always retained."""
        self._branches[branch.id] = branch

    def get_branch(self, branch_id: str) -> Optional[PortfolioBranch]:
        return self._branches.get(branch_id)

    def list_branches(self, state: Optional[BranchState] = None) -> List[PortfolioBranch]:
        if state is None:
            return list(self._branches.values())
        return [b for b in self._branches.values() if b.branch_state == state]

    def mark_branch_failed(self, branch_id: str, failure_reason: str) -> None:
        """Mark a branch as failed while preserving it permanently in history."""
        b = self._branches.get(branch_id)
        if b is None:
            raise KeyError(f"Branch '{branch_id}' not found.")
        updated = PortfolioBranch(
            id=b.id,
            schema_version=b.schema_version,
            originating_hypothesis_id=b.originating_hypothesis_id,
            action=b.action,
            score=b.score,
            expected_value=b.expected_value,
            resource_allocation=0.0,
            diversity_features=b.diversity_features,
            branch_state=BranchState.FAILED,
            history_experiment_ids=b.history_experiment_ids,
            failure_history=b.failure_history + [failure_reason],
            provenance_id=b.provenance_id,
            metadata=b.metadata,
        )
        self._branches[branch_id] = updated

    def compute_diversity(self) -> Dict[str, float]:
        """Compute exposed diversity metrics across active branches.
        
        Metrics:
        - strategy_entropy: Shannon entropy across strategy families
        - model_family_entropy: Shannon entropy across model families
        - total_branches: total count
        - active_branches: active count
        - failed_branches_retained: count of preserved negative branches
        """
        active = self.list_branches(BranchState.ACTIVE)
        if not active:
            return {
                "strategy_entropy": 0.0,
                "model_family_entropy": 0.0,
                "total_branches": len(self._branches),
                "active_branches": 0,
                "failed_branches_retained": len(self.list_branches(BranchState.FAILED)),
            }

        def _entropy(items: List[str]) -> float:
            if not items:
                return 0.0
            counts: Dict[str, int] = {}
            for item in items:
                counts[item] = counts.get(item, 0) + 1
            n = len(items)
            return -sum((c / n) * math.log2(c / n) for c in counts.values())

        strategies = [b.diversity_features.get("strategy_family", "unknown") for b in active]
        models = [b.diversity_features.get("model_family", "unknown") for b in active]

        return {
            "strategy_entropy": round(_entropy(strategies), 4),
            "model_family_entropy": round(_entropy(models), 4),
            "total_branches": len(self._branches),
            "active_branches": len(active),
            "failed_branches_retained": len(self.list_branches(BranchState.FAILED)),
        }

    def allocate_resources(self, total_budget: float) -> Dict[str, float]:
        """Proportionally allocate budget across active branches based on scores."""
        active = self.list_branches(BranchState.ACTIVE)
        if not active:
            return {}

        min_score = min(b.score for b in active)
        # Shift scores so all are positive
        shifted = [b.score - min_score + 0.1 for b in active]
        total_score = sum(shifted)

        allocations: Dict[str, float] = {}
        for b, s in zip(active, shifted):
            frac = s / total_score
            allocations[b.id] = round(frac * total_budget, 4)
        return allocations
