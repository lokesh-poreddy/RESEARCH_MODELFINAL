"""researchforge/router/modes.py — ResearchMode enum and ModeCapabilities definitions.

Phase 13 (RF-1.0.0-beta.1):
Codifies the 9 canonical research modes planned in RESEARCHFORGE_STATE.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ResearchMode(str, Enum):
    """The 9 canonical operational research modes."""
    FAST_EVIDENCE = "FAST_EVIDENCE"        # Rapid preliminary probing with conservative compute
    DEEP_RESEARCH = "DEEP_RESEARCH"        # Exhaustive exploration with literature and deep memory retrieval
    EXPERIMENTAL = "EXPERIMENTAL"          # High-entropy exploratory mutation and novel candidate search
    REPLICATION = "REPLICATION"            # Confirmatory reproduction of prior high-performing candidates
    CHALLENGE = "CHALLENGE"                # Adversarial stress-testing, label perturbation, data shift
    TRANSFER = "TRANSFER"                  # Cross-domain capability projection guarded by TransferGuard
    SIMULATION = "SIMULATION"              # Surrogate-model evaluation under strict budget constraints
    BACKGROUND = "BACKGROUND"              # Asynchronous consolidation, memory pruning, retrospective analysis
    HUMAN_REVIEW = "HUMAN_REVIEW"          # Safety/governance pause awaiting external confirmation


@dataclass(frozen=True)
class ModeCapabilities:
    """Operational constraints and algorithmic capabilities enabled in a specific mode."""
    mode: ResearchMode
    max_trials_per_cycle: int
    timeout_scale: float
    exploration_temperature: float
    retrieval_depth: int
    transfer_guard_required: bool
    allows_mutation: bool
    allows_crossover: bool
    description: str


MODE_CAPABILITIES: Dict[ResearchMode, ModeCapabilities] = {
    ResearchMode.FAST_EVIDENCE: ModeCapabilities(
        mode=ResearchMode.FAST_EVIDENCE,
        max_trials_per_cycle=3,
        timeout_scale=0.5,
        exploration_temperature=0.3,
        retrieval_depth=2,
        transfer_guard_required=False,
        allows_mutation=True,
        allows_crossover=False,
        description="Fast evaluation with minimal overhead.",
    ),
    ResearchMode.DEEP_RESEARCH: ModeCapabilities(
        mode=ResearchMode.DEEP_RESEARCH,
        max_trials_per_cycle=10,
        timeout_scale=1.5,
        exploration_temperature=0.8,
        retrieval_depth=10,
        transfer_guard_required=False,
        allows_mutation=True,
        allows_crossover=True,
        description="Full exploration with deep literature and memory context.",
    ),
    ResearchMode.EXPERIMENTAL: ModeCapabilities(
        mode=ResearchMode.EXPERIMENTAL,
        max_trials_per_cycle=8,
        timeout_scale=1.0,
        exploration_temperature=1.5,
        retrieval_depth=4,
        transfer_guard_required=False,
        allows_mutation=True,
        allows_crossover=True,
        description="High-entropy search prioritizing novelty and radical mutation.",
    ),
    ResearchMode.REPLICATION: ModeCapabilities(
        mode=ResearchMode.REPLICATION,
        max_trials_per_cycle=5,
        timeout_scale=1.0,
        exploration_temperature=0.0,
        retrieval_depth=5,
        transfer_guard_required=False,
        allows_mutation=False,
        allows_crossover=False,
        description="Deterministic re-execution to verify stability across seeds.",
    ),
    ResearchMode.CHALLENGE: ModeCapabilities(
        mode=ResearchMode.CHALLENGE,
        max_trials_per_cycle=6,
        timeout_scale=1.2,
        exploration_temperature=0.9,
        retrieval_depth=4,
        transfer_guard_required=False,
        allows_mutation=True,
        allows_crossover=False,
        description="Stress testing against edge cases and distribution shifts.",
    ),
    ResearchMode.TRANSFER: ModeCapabilities(
        mode=ResearchMode.TRANSFER,
        max_trials_per_cycle=6,
        timeout_scale=1.0,
        exploration_temperature=0.5,
        retrieval_depth=8,
        transfer_guard_required=True,
        allows_mutation=True,
        allows_crossover=True,
        description="Cross-domain transfer explicitly mediated by TransferGuard.",
    ),
    ResearchMode.SIMULATION: ModeCapabilities(
        mode=ResearchMode.SIMULATION,
        max_trials_per_cycle=15,
        timeout_scale=0.2,
        exploration_temperature=0.7,
        retrieval_depth=2,
        transfer_guard_required=False,
        allows_mutation=True,
        allows_crossover=True,
        description="Low-cost surrogate model evaluation.",
    ),
    ResearchMode.BACKGROUND: ModeCapabilities(
        mode=ResearchMode.BACKGROUND,
        max_trials_per_cycle=0,
        timeout_scale=0.0,
        exploration_temperature=0.0,
        retrieval_depth=10,
        transfer_guard_required=False,
        allows_mutation=False,
        allows_crossover=False,
        description="Offline memory consolidation and provenance graph indexing.",
    ),
    ResearchMode.HUMAN_REVIEW: ModeCapabilities(
        mode=ResearchMode.HUMAN_REVIEW,
        max_trials_per_cycle=0,
        timeout_scale=0.0,
        exploration_temperature=0.0,
        retrieval_depth=0,
        transfer_guard_required=False,
        allows_mutation=False,
        allows_crossover=False,
        description="Execution suspended awaiting explicit human authorization.",
    ),
}
