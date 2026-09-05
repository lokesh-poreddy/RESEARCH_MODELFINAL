"""researchforge/router/transitions.py — Auditable mode transition events.

Phase 13 (RF-1.0.0-beta.1):
Enforces Criterion 5: Every mode transition records state_before, state_after,
trigger, trigger_value, threshold, policy_version, available_actions,
selected_action, and alternative_actions with cryptographic fingerprinting.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModeTransitionEvent:
    """Full-provenance event recording an evidence-triggered research mode transition."""
    state_before: str
    state_after: str
    trigger: str
    trigger_value: Any
    threshold: Any
    policy_version: str
    available_actions: List[str]
    selected_action: str
    alternative_actions: List[str]
    rationale: str
    timestamp: float = field(default_factory=time.time)
    event_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.event_fingerprint:
            payload = {
                "state_before": self.state_before,
                "state_after": self.state_after,
                "trigger": self.trigger,
                "trigger_value": str(self.trigger_value),
                "threshold": str(self.threshold),
                "policy_version": self.policy_version,
                "selected_action": self.selected_action,
                "available_actions": sorted(self.available_actions),
                "alternative_actions": sorted(self.alternative_actions),
            }
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            object.__setattr__(self, "event_fingerprint", hashlib.sha256(raw).hexdigest()[:16])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_before": self.state_before,
            "state_after": self.state_after,
            "trigger": self.trigger,
            "trigger_value": self.trigger_value,
            "threshold": self.threshold,
            "policy_version": self.policy_version,
            "available_actions": list(self.available_actions),
            "selected_action": self.selected_action,
            "alternative_actions": list(self.alternative_actions),
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "event_fingerprint": self.event_fingerprint,
        }
