"""researchforge/router/ — Dynamic Mode Router and research mode scheduling.

Phase 13 (RF-1.0.0-beta.1):
Evidence-triggered deterministic state machine managing the 9 canonical research modes.
"""
from .modes import ResearchMode, ModeCapabilities, MODE_CAPABILITIES
from .transitions import ModeTransitionEvent
from .router import DynamicModeRouter

__all__ = [
    "ResearchMode",
    "ModeCapabilities",
    "MODE_CAPABILITIES",
    "ModeTransitionEvent",
    "DynamicModeRouter",
]
