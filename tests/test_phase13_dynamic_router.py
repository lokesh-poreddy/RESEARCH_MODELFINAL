"""tests/test_phase13_dynamic_router.py — Dynamic Mode Router tests.

Classification: CORE
Verifies:
1. All 9 canonical research modes are represented with valid ModeCapabilities.
2. State transitions are deterministic and evidence-triggered.
3. Every transition logs all 9 provenance metadata fields.
4. Typed SaturationReport signals drive policy-governed mode switches.
5. Resource budget exhaustion triggers conservation mode (FAST_EVIDENCE).
"""
import pytest
from researchforge.policy.saturation import PivotRecommendation, SaturationReport, SaturationState
from researchforge.router.modes import MODE_CAPABILITIES, ResearchMode
from researchforge.router.router import DynamicModeRouter
from researchforge.router.transitions import ModeTransitionEvent


def test_all_9_modes_represented():
    """All 9 modes defined in RESEARCHFORGE_STATE.yaml must be present with valid capabilities."""
    expected_modes = {
        ResearchMode.FAST_EVIDENCE,
        ResearchMode.DEEP_RESEARCH,
        ResearchMode.EXPERIMENTAL,
        ResearchMode.REPLICATION,
        ResearchMode.CHALLENGE,
        ResearchMode.TRANSFER,
        ResearchMode.SIMULATION,
        ResearchMode.BACKGROUND,
        ResearchMode.HUMAN_REVIEW,
    }
    assert set(ResearchMode) == expected_modes
    assert set(MODE_CAPABILITIES.keys()) == expected_modes

    for mode, caps in MODE_CAPABILITIES.items():
        assert caps.mode == mode
        assert caps.max_trials_per_cycle >= 0
        assert caps.timeout_scale >= 0.0
        assert isinstance(caps.description, str) and len(caps.description) > 0


def test_saturation_event_triggers_policy_transition():
    """SaturationReport recommending CHANGE_STRATEGY must deterministically switch to EXPERIMENTAL mode."""
    router = DynamicModeRouter(initial_mode=ResearchMode.DEEP_RESEARCH)
    report = SaturationReport(
        id="sat_router_01",
        schema_version="1.0",
        current_status=SaturationState.SATURATED,
        recommendation=PivotRecommendation.CHANGE_STRATEGY,
        signal_values={"frr": 0.8},
        thresholds={"frr_max": 0.5},
        triggering_evidence=["3 consecutive failure trials"],
        rationale="Hypothesis search stagnant",
    )

    event = router.handle_saturation_report(report)

    assert event is not None
    assert router.current_mode == ResearchMode.EXPERIMENTAL
    assert event.state_before == "DEEP_RESEARCH"
    assert event.state_after == "EXPERIMENTAL"
    assert event.trigger == "SATURATION_DETECTED"
    assert event.trigger_value == "SATURATED"
    assert event.threshold == "SATURATED"
    assert event.selected_action == "EXPERIMENTAL"
    assert len(event.available_actions) == 9
    assert len(event.alternative_actions) == 8
    assert len(event.event_fingerprint) == 16


def test_budget_exhaustion_warning_triggers_fast_evidence():
    """Budget falling below 15% should trigger conservation via FAST_EVIDENCE."""
    router = DynamicModeRouter(initial_mode=ResearchMode.DEEP_RESEARCH)

    # 10 remaining out of 100 trials = 10%
    event = router.handle_budget_signal(remaining_budget=10, total_budget=100)

    assert event is not None
    assert router.current_mode == ResearchMode.FAST_EVIDENCE
    assert event.state_after == "FAST_EVIDENCE"
    assert event.trigger == "BUDGET_EXHAUSTION_WARNING"
    assert event.trigger_value == 0.10


def test_cross_family_transfer_context_triggers_transfer_mode():
    """Entering a cross-family task with prior experience triggers TRANSFER mode."""
    router = DynamicModeRouter(initial_mode=ResearchMode.DEEP_RESEARCH)

    event = router.handle_transfer_context(is_cross_family=True, prior_task_count=2)

    assert event is not None
    assert router.current_mode == ResearchMode.TRANSFER
    assert event.state_after == "TRANSFER"
    assert event.trigger == "CROSS_FAMILY_TRANSFER_DETECTED"
    assert event.trigger_value == 2
