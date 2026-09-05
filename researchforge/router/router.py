"""researchforge/router/router.py — Deterministic Dynamic Mode Router.

Phase 13 (RF-1.0.0-beta.1):
Implements evidence-triggered deterministic state machine across the 9 research modes.
Every transition is logged with complete analytical provenance.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..policy.saturation import PivotRecommendation, SaturationReport, SaturationState
from .modes import MODE_CAPABILITIES, ModeCapabilities, ResearchMode
from .transitions import ModeTransitionEvent

logger = logging.getLogger(__name__)

POLICY_VERSION = "1.0.0-beta.1"


class DynamicModeRouter:
    """Orchestrates research mode transitions based on typed environmental and research signals."""

    def __init__(
        self,
        initial_mode: ResearchMode = ResearchMode.DEEP_RESEARCH,
        policy_version: str = POLICY_VERSION,
    ) -> None:
        self._current_mode = initial_mode
        self._policy_version = policy_version
        self._transition_history: List[ModeTransitionEvent] = []

    @property
    def current_mode(self) -> ResearchMode:
        return self._current_mode

    @property
    def capabilities(self) -> ModeCapabilities:
        return MODE_CAPABILITIES[self._current_mode]

    @property
    def transition_history(self) -> List[ModeTransitionEvent]:
        return list(self._transition_history)

    def handle_saturation_report(self, report: SaturationReport) -> Optional[ModeTransitionEvent]:
        """Trigger mode transition if research saturation detector recommends a pivot."""
        target_mode: Optional[ResearchMode] = None
        rat: str = ""

        if report.current_status in (SaturationState.SATURATED, SaturationState.STAGNANT):
            if report.recommendation == PivotRecommendation.CHANGE_STRATEGY:
                target_mode = ResearchMode.EXPERIMENTAL
                rat = f"Saturation detected ({report.current_status.value}): pivoting to high-entropy EXPERIMENTAL mode."
            elif report.recommendation == PivotRecommendation.REPLICATE:
                target_mode = ResearchMode.REPLICATION
                rat = f"Confirmation required ({report.current_status.value}): switching to deterministic REPLICATION mode."
            elif report.recommendation == PivotRecommendation.REQUEST_HUMAN_REVIEW:
                target_mode = ResearchMode.HUMAN_REVIEW
                rat = f"Escalation required ({report.current_status.value}): requesting HUMAN_REVIEW."
            elif report.recommendation == PivotRecommendation.GATHER_EVIDENCE:
                target_mode = ResearchMode.DEEP_RESEARCH
                rat = f"Information deficit ({report.current_status.value}): switching to DEEP_RESEARCH mode."

        if target_mode is not None and target_mode != self._current_mode:
            all_modes = [m.value for m in ResearchMode]
            event = ModeTransitionEvent(
                state_before=self._current_mode.value,
                state_after=target_mode.value,
                trigger="SATURATION_DETECTED",
                trigger_value=report.current_status.value,
                threshold=SaturationState.SATURATED.value,
                policy_version=self._policy_version,
                available_actions=all_modes,
                selected_action=target_mode.value,
                alternative_actions=[m for m in all_modes if m != target_mode.value],
                rationale=rat,
            )
            self._current_mode = target_mode
            self._transition_history.append(event)
            return event

        return None

    def handle_budget_signal(self, remaining_budget: int, total_budget: int) -> Optional[ModeTransitionEvent]:
        """Trigger transition to FAST_EVIDENCE when compute budget is severely constrained."""
        if total_budget <= 0:
            return None

        ratio = remaining_budget / total_budget
        threshold = 0.15

        if ratio <= threshold and self._current_mode in (ResearchMode.DEEP_RESEARCH, ResearchMode.EXPERIMENTAL):
            target_mode = ResearchMode.FAST_EVIDENCE
            all_modes = [m.value for m in ResearchMode]
            rat = f"Budget ratio {ratio:.2f} <= {threshold:.2f}: conserving resources via FAST_EVIDENCE."
            event = ModeTransitionEvent(
                state_before=self._current_mode.value,
                state_after=target_mode.value,
                trigger="BUDGET_EXHAUSTION_WARNING",
                trigger_value=round(ratio, 4),
                threshold=threshold,
                policy_version=self._policy_version,
                available_actions=all_modes,
                selected_action=target_mode.value,
                alternative_actions=[m for m in all_modes if m != target_mode.value],
                rationale=rat,
            )
            self._current_mode = target_mode
            self._transition_history.append(event)
            return event

        return None

    def handle_transfer_context(self, is_cross_family: bool, prior_task_count: int) -> Optional[ModeTransitionEvent]:
        """Trigger transition to TRANSFER mode when cross-family prior experience is detected."""
        if is_cross_family and prior_task_count > 0 and self._current_mode != ResearchMode.TRANSFER:
            target_mode = ResearchMode.TRANSFER
            all_modes = [m.value for m in ResearchMode]
            rat = f"Cross-family transfer detected with {prior_task_count} prior tasks: engaging TRANSFER mode with TransferGuard."
            event = ModeTransitionEvent(
                state_before=self._current_mode.value,
                state_after=target_mode.value,
                trigger="CROSS_FAMILY_TRANSFER_DETECTED",
                trigger_value=prior_task_count,
                threshold=1,
                policy_version=self._policy_version,
                available_actions=all_modes,
                selected_action=target_mode.value,
                alternative_actions=[m for m in all_modes if m != target_mode.value],
                rationale=rat,
            )
            self._current_mode = target_mode
            self._transition_history.append(event)
            return event

        return None

    def force_transition(self, target_mode: ResearchMode, reason: str) -> ModeTransitionEvent:
        """Explicit programmatic transition with full provenance audit."""
        all_modes = [m.value for m in ResearchMode]
        event = ModeTransitionEvent(
            state_before=self._current_mode.value,
            state_after=target_mode.value,
            trigger="EXPLICIT_TRANSITION",
            trigger_value=target_mode.value,
            threshold=None,
            policy_version=self._policy_version,
            available_actions=all_modes,
            selected_action=target_mode.value,
            alternative_actions=[m for m in all_modes if m != target_mode.value],
            rationale=reason,
        )
        self._current_mode = target_mode
        self._transition_history.append(event)
        return event
