"""researchforge/domain/action.py — Canonical ResearchAction domain contract.

Scientific Objective (Phase 9):
Defines explicit, candidate research actions that a ResearchPolicy can evaluate,
rank, and select from during autonomous research loop execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .base import DomainObject


class ActionType(str, Enum):
    """Canonical research action categories."""
    EXPLOIT_KNOWN_STRATEGY = "EXPLOIT_KNOWN_STRATEGY"
    EXPLORE_NEW_STRATEGY = "EXPLORE_NEW_STRATEGY"
    REPLICATE = "REPLICATE"
    MODIFY_MODEL = "MODIFY_MODEL"
    COMBINE_COMPONENTS = "COMBINE_COMPONENTS"
    RETRY_AFTER_FAILURE = "RETRY_AFTER_FAILURE"
    SEARCH_EVIDENCE = "SEARCH_EVIDENCE"
    TEST_HYPOTHESIS = "TEST_HYPOTHESIS"
    TRANSFER = "TRANSFER"
    PIVOT = "PIVOT"


@dataclass(frozen=True)
class ResearchAction(DomainObject):
    """Canonical frozen representation of an autonomous research action."""
    action_type: ActionType | str
    target_context: Dict[str, Any]
    expected_objective: str
    parameters: Dict[str, Any] | None = None
    provenance_id: str | None = None
    configuration_fingerprint: str | None = None
    metadata: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ResearchAction":
        d = dict(obj)
        if "action_type" in d and isinstance(d["action_type"], str):
            try:
                d["action_type"] = ActionType(d["action_type"])
            except ValueError:
                pass  # Keep as string for extensible future actions
        return cls(**d)
