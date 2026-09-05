"""researchforge/repositories/replay.py — Deterministic event replay service.

RF-1.0.0-alpha.3 (Phase 8A): Replays append-only research events to reconstruct exact ResearchState and VRDEG.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..domain.provenance import Provenance
from ..domain.state import ResearchState
from ..state.events import Event, EventType
from ..state.transition_engine import ResearchStateTransitionEngine
from ..vrdeg.graph import VRDEG
from ..vrdeg.projector import VRDEGProjector
from .event_store import ResearchEventRecord


class EventReplayService:
    """Replays chronological research events from the event store to reconstruct state and graph."""

    def __init__(
        self,
        engine: Optional[ResearchStateTransitionEngine] = None,
    ) -> None:
        self.engine = engine or ResearchStateTransitionEngine()

    def replay_trajectory(
        self,
        initial_state: ResearchState,
        events: List[ResearchEventRecord],
        target_graph: Optional[VRDEG] = None,
    ) -> Tuple[ResearchState, VRDEG]:
        """Reconstruct the resulting ResearchState and VRDEG graph deterministically."""
        current_state = initial_state
        graph = target_graph if target_graph is not None else VRDEG()
        projector = VRDEGProjector(graph)

        # Sort strictly by global sequence then aggregate sequence to ensure deterministic ordering without timestamps
        sorted_events = sorted(events, key=lambda e: (e.global_sequence if e.global_sequence is not None else 0, e.sequence_number))

        for rec in sorted_events:
            event_obj = Event(
                id=rec.id,
                schema_version=rec.schema_version,
                event_type=rec.event_type,
                payload=rec.payload,
                timestamp=str(rec.created_at) if rec.created_at else None,
                provenance_id=rec.provenance_id,
            )

            prov = None
            if rec.provenance_id:
                prov = Provenance(
                    id=rec.provenance_id,
                    schema_version="1.0",
                    created_by="event_replay",
                    created_at=str(rec.created_at or 0.0),
                )

            # 1. State transition
            try:
                current_state = self.engine.transition(current_state, event_obj, provenance=prov)
            except Exception:
                # If transition is non-state-mutating event, proceed to projection
                pass

            # 2. Graph projection
            try:
                projector.project_event(event_obj)
            except Exception:
                pass

        return current_state, graph

    def verify_determinism(
        self,
        initial_state: ResearchState,
        events: List[ResearchEventRecord],
        expected_state_fingerprint: str,
    ) -> bool:
        """Verify that replaying events yields the exact expected state fingerprint."""
        final_state, _ = self.replay_trajectory(initial_state, events)
        return final_state.fingerprint() == expected_state_fingerprint


class PolicyDecisionReplayService:
    """Deterministic replay service for ResearchPolicy decisions.

    Verifies:
    ResearchState + EvidenceSnapshot + PolicyConfig (PolicyVersion) + candidate actions
    -> replayed policy decision with bitwise identical decision fingerprint.
    """

    def replay_decision(
        self,
        state: ResearchState,
        candidate_actions: List[Any],
        snapshot: Any,
        policy_config: Any,
        decision_timestamp: float,
    ) -> Tuple[Any, Any]:
        from ..policy.research_policy import ResearchPolicy
        policy = ResearchPolicy(config=policy_config)
        return policy.evaluate_candidates(
            state=state,
            candidate_actions=candidate_actions,
            snapshot=snapshot,
            decision_timestamp=decision_timestamp,
        )

    def verify_decision_determinism(
        self,
        original_record: Any,
        state: ResearchState,
        candidate_actions: List[Any],
        snapshot: Any,
        policy_config: Any,
    ) -> bool:
        _, replayed_record = self.replay_decision(
            state=state,
            candidate_actions=candidate_actions,
            snapshot=snapshot,
            policy_config=policy_config,
            decision_timestamp=original_record.decision_timestamp,
        )
        return replayed_record.decision_fingerprint() == original_record.decision_fingerprint()

    def reconstruct_observation(
        self,
        policy_decision: Any,
        spec_id: str,
        run_id: str,
        outcome_id: str,
        actual_metric: Optional[float],
        baseline_metric: Optional[float] = None,
        validity_verdict: Any = None,
        evaluator_basis: str = "StandardScientificEvaluator",
        observation_timestamp: Optional[Any] = None,
    ) -> Any:
        from ..domain.validity import ValidityVerdict
        from ..policy.telemetry import ResearchOutcomeObservation
        verdict = validity_verdict or ValidityVerdict.PASS
        return ResearchOutcomeObservation.create(
            policy_decision=policy_decision,
            spec_id=spec_id,
            run_id=run_id,
            outcome_id=outcome_id,
            actual_metric=actual_metric,
            baseline_metric=baseline_metric,
            validity_verdict=verdict,
            evaluator_basis=evaluator_basis,
            observation_timestamp=observation_timestamp,
        )

