"""tests/test_phase9a_telemetry.py — Phase 9A Policy Observation & Research Telemetry Freeze.

Requirements Tested (Phase 9A Instructions):
1. Complete Telemetry Schema: All 18-20 telemetry properties are schema-addressable.
2. Optional/Unavailable Handling: Fields without observations are None / NOT_APPLICABLE, never fabricated.
3. Observational Progress Output: Evaluator produces VALID_EMPIRICAL_GAIN, NO_CHANGE, DEGRADATION,
   INVALID, INCONCLUSIVE, or NOT_ASSESSED without asserting causal claims.
4. Downstream Outcome Binding: Connects predicted_utility -> actual -> baseline -> delta -> validity -> progress.
5. Deterministic Decision & Observation Replay: Identical inputs reproduce bitwise-identical fingerprints.
6. Timezone-Aware UTC Timestamping: Verified datetime and ISO-8601 formatting.
"""
from datetime import datetime, timezone
import pytest

from researchforge.domain.action import ActionType, ResearchAction
from researchforge.domain.state import ResearchState
from researchforge.domain.validity import ValidityVerdict
from researchforge.evidence.snapshot import EvidenceSnapshot
from researchforge.policy.config import PolicyConfig
from researchforge.policy.decision_record import PolicyDecisionRecord
from researchforge.policy.research_policy import ResearchPolicy
from researchforge.policy.score_decomposition import ScoreDecomposition
from researchforge.policy.telemetry import ResearchOutcomeObservation, ResearchProgressStatus
from researchforge.repositories.replay import PolicyDecisionReplayService


# ==============================================================================
# 1. Complete Telemetry Schema & Addressability
# ==============================================================================

def test_telemetry_schema_addressability_and_export():
    """All 18+ required telemetry fields are schema-addressable on PolicyDecisionRecord."""
    policy = ResearchPolicy(config=PolicyConfig.baseline_v1())
    state = ResearchState(id="s_tel_01", schema_version="1.0", problem_id="prob_01")
    candidates = [
        ResearchAction(
            id="act_01",
            schema_version="1.0",
            action_type=ActionType.EXPLORE_NEW_STRATEGY,
            target_context={"strategy": "adamw", "model_family": "mlp"},
            expected_objective="Objective 1",
            provenance_id="prov_act_01",
        ),
        ResearchAction(
            id="act_02",
            schema_version="1.0",
            action_type=ActionType.EXPLOIT_KNOWN_STRATEGY,
            target_context={"strategy": "sgd", "model_family": "mlp"},
            expected_objective="Objective 2",
            provenance_id="prov_act_02",
        ),
    ]
    snapshot = EvidenceSnapshot.create(decision_id="dec_tel_01", as_of_timestamp=100.0, evidence_ids=[])

    dec, record = policy.evaluate_candidates(state, candidates, snapshot, decision_timestamp=100.0)

    # 1. ResearchState fingerprint
    assert record.research_state_fingerprint == state.fingerprint()
    # 2. EvidenceSnapshot ID
    assert record.evidence_snapshot_id == snapshot.id
    # 3. EvidenceSnapshot fingerprint
    assert record.evidence_snapshot_fingerprint is not None
    # 4. Memory retrieval summary
    assert hasattr(record, "memory_retrieval_summary")
    # 5. Candidate actions
    assert len(record.candidate_actions) == 2
    # 6. Candidate component scores
    assert "act_01" in record.candidate_component_scores
    assert "performance" in record.candidate_component_scores["act_01"]
    # 7. Weighted score contributions
    assert "w_performance" in record.weighted_score_contributions["act_01"]
    # 8. Policy version
    assert record.policy_version == "baseline_v1"
    # 9. Policy fingerprint
    assert len(record.policy_fingerprint) == 64
    # 10. Portfolio state
    assert hasattr(record, "portfolio_state")
    # 11. Selected action
    assert record.selected_action.id == record.selected_action_id
    # 12. Selected action ID
    assert record.selected_action_id in ["act_01", "act_02"]
    # 13. Rejected alternatives
    assert len(record.rejected_alternatives) == 1
    # 14. Predicted utility
    assert isinstance(record.predicted_utility, float)
    # 15. Baseline score
    assert record.baseline_score is not None
    # 16. Memory contribution
    assert record.memory_contribution is not None
    # 17. Evidence contribution
    assert record.evidence_contribution is not None
    # 18. Failure-risk contribution
    assert record.failure_risk_contribution is not None
    # 19. Expected cost
    assert record.expected_cost is not None
    # 20. Provenance ID
    assert record.provenance_id is not None

    # Verify telemetry_dict produces non-empty dict containing all fields
    t_dict = record.telemetry_dict()
    assert len(t_dict) >= 20
    assert t_dict["policy_version"] == "baseline_v1"
    assert t_dict["predicted_utility"] == record.predicted_utility


def test_optional_unavailable_telemetry_is_not_fabricated():
    """Fields without observations are None or empty, never fabricated with dummy numbers."""
    act = ResearchAction(
        id="act_bare",
        schema_version="1.0",
        action_type=ActionType.PIVOT,
        target_context={},
        expected_objective="Bare test",
    )
    decomp = ScoreDecomposition.compute(
        action_id=act.id,
        action_type=act.action_type.value,
        raw_scores={},
        weights={},
    )
    # Record without optional fields provided
    bare_record = PolicyDecisionRecord(
        id="pdr_bare",
        schema_version="1.0",
        decision_id="dec_bare",
        research_state_fingerprint="fp_bare",
        evidence_snapshot_id="snap_bare",
        policy_version="baseline_v1",
        policy_fingerprint="fp_pol_bare",
        selected_action_id=act.id,
        selected_action=act,
        rejected_action_ids=[],
        score_decompositions={act.id: decomp},
        policy_weights={},
        decision_timestamp=150.0,
        explanation="Bare explanation",
        available_memory_ref=None,
        provenance_id=None,
        credit_assignment_meta=None,
        portfolio_state=None,
        baseline_score=None,
    )

    assert bare_record.available_memory_ref is None
    assert bare_record.portfolio_state is None
    assert bare_record.baseline_score is None
    assert bare_record.provenance_id is None
    # Telemetry dict honors unobserved fields as None
    t_dict = bare_record.telemetry_dict()
    assert t_dict["portfolio_state"] is None
    assert t_dict["baseline_score"] is None


# ==============================================================================
# 2. Timezone-Aware UTC Timestamping
# ==============================================================================

def test_timezone_aware_utc_timestamps():
    """Timestamps are strictly timezone-aware UTC datetime and ISO-8601 strings."""
    policy = ResearchPolicy(config=PolicyConfig.baseline_v1())
    state = ResearchState(id="s_time", schema_version="1.0")
    act = ResearchAction(id="a_time", schema_version="1.0", action_type=ActionType.MODIFY_MODEL, target_context={}, expected_objective="Time")
    snapshot = EvidenceSnapshot.create(decision_id="d_time", as_of_timestamp=1700000000.0, evidence_ids=[])

    _, record = policy.evaluate_candidates(state, [act], snapshot, decision_timestamp=1700000000.0)

    # datetime is timezone-aware UTC
    dt = record.decision_datetime
    assert dt.tzinfo == timezone.utc
    assert record.decision_timestamp_iso.startswith("2023-11-14T")
    assert "+00:00" in record.decision_timestamp_iso or "Z" in record.decision_timestamp_iso


# ==============================================================================
# 3. Downstream Outcome Observation Binding & Scientific Validity Separation
# ==============================================================================

def test_outcome_observation_binding_and_progress_classification():
    """Binds predicted utility -> actual outcome -> baseline -> delta -> validity -> progress."""
    act = ResearchAction(id="act_obs", schema_version="1.0", action_type=ActionType.EXPLORE_NEW_STRATEGY, target_context={}, expected_objective="Obs")
    decomp = ScoreDecomposition.compute(
        action_id=act.id,
        action_type=act.action_type.value,
        raw_scores={"performance": 0.85},
        weights={"w_performance": 1.0},
    )
    record = PolicyDecisionRecord(
        id="pdr_obs_01",
        schema_version="1.0",
        decision_id="dec_obs_01",
        research_state_fingerprint="fp_s",
        evidence_snapshot_id="snap_s",
        policy_version="baseline_v1",
        policy_fingerprint="fp_p",
        selected_action_id=act.id,
        selected_action=act,
        rejected_action_ids=[],
        score_decompositions={act.id: decomp},
        policy_weights={"w_performance": 1.0},
        decision_timestamp=200.0,
        explanation="Test obs",
        provenance_id="prov_01",
    )

    # Case A: Valid empirical gain over baseline
    obs_gain = ResearchOutcomeObservation.create(
        policy_decision=record,
        spec_id="spec_01",
        run_id="run_01",
        outcome_id="out_01",
        actual_metric=0.88,
        baseline_metric=0.80,
        validity_verdict=ValidityVerdict.PASS,
        evaluator_basis="StandardScientificEvaluator",
    )
    assert obs_gain.research_progress_status == ResearchProgressStatus.VALID_EMPIRICAL_GAIN
    assert obs_gain.delta == pytest.approx(0.08, 1e-4)
    assert obs_gain.research_progress_value == pytest.approx(0.08, 1e-4)
    assert "valid empirical gain" in obs_gain.research_progress_basis

    # Case B: Empirical degradation below baseline
    obs_deg = ResearchOutcomeObservation.create(
        policy_decision=record,
        spec_id="spec_01",
        run_id="run_01",
        outcome_id="out_02",
        actual_metric=0.75,
        baseline_metric=0.80,
        validity_verdict=ValidityVerdict.PASS,
        evaluator_basis="StandardScientificEvaluator",
    )
    assert obs_deg.research_progress_status == ResearchProgressStatus.DEGRADATION
    assert obs_deg.delta == pytest.approx(-0.05, 1e-4)
    assert obs_deg.research_progress_value == pytest.approx(-0.05, 1e-4)

    # Case C: Empirical parity / no change
    obs_parity = ResearchOutcomeObservation.create(
        policy_decision=record,
        spec_id="spec_01",
        run_id="run_01",
        outcome_id="out_03",
        actual_metric=0.80,
        baseline_metric=0.80,
        validity_verdict=ValidityVerdict.PASS,
        evaluator_basis="StandardScientificEvaluator",
    )
    assert obs_parity.research_progress_status == ResearchProgressStatus.NO_CHANGE
    assert obs_parity.research_progress_value == 0.0

    # Case D: Validity check failed -> Status INVALID regardless of nominal score gain!
    obs_invalid = ResearchOutcomeObservation.create(
        policy_decision=record,
        spec_id="spec_01",
        run_id="run_01",
        outcome_id="out_04",
        actual_metric=0.99,  # High nominal metric
        baseline_metric=0.80,
        validity_verdict=ValidityVerdict.FAIL,  # Invalid run!
        evaluator_basis="StandardScientificEvaluator",
    )
    assert obs_invalid.research_progress_status == ResearchProgressStatus.INVALID
    assert obs_invalid.research_progress_value is None  # Never certify invalid gain
    assert "validity verdict is FAIL" in obs_invalid.research_progress_basis

    # Case E: Missing baseline -> Status NOT_ASSESSED
    obs_nobase = ResearchOutcomeObservation.create(
        policy_decision=record,
        spec_id="spec_01",
        run_id="run_01",
        outcome_id="out_05",
        actual_metric=0.85,
        baseline_metric=None,
        validity_verdict=ValidityVerdict.PASS,
        evaluator_basis="StandardScientificEvaluator",
    )
    assert obs_nobase.research_progress_status == ResearchProgressStatus.NOT_ASSESSED
    assert obs_nobase.research_progress_value is None


# ==============================================================================
# 4. Deterministic Replay of Policy Decisions and Outcome Observations
# ==============================================================================

def test_deterministic_decision_and_observation_replay():
    """PolicyDecision and ResearchOutcomeObservation are bitwise reconstructable upon replay."""
    service = PolicyDecisionReplayService()
    cfg = PolicyConfig.baseline_v1()
    policy = ResearchPolicy(config=cfg)
    state = ResearchState(id="s_rep_9a", schema_version="1.0")
    cand = [
        ResearchAction(
            id="act_rep_1",
            schema_version="1.0",
            action_type=ActionType.EXPLORE_NEW_STRATEGY,
            target_context={"strategy": "adamw", "model_family": "mlp"},
            expected_objective="Replay test",
        )
    ]
    snapshot = EvidenceSnapshot.create(decision_id="dec_rep_9a", as_of_timestamp=100.0, evidence_ids=[])

    _, original_record = policy.evaluate_candidates(state, cand, snapshot, decision_timestamp=100.0)

    # 1. Verify policy decision replay determinism
    is_det = service.verify_decision_determinism(
        original_record=original_record,
        state=state,
        candidate_actions=cand,
        snapshot=snapshot,
        policy_config=cfg,
    )
    assert is_det is True

    # 2. Construct original observation
    obs_original = service.reconstruct_observation(
        policy_decision=original_record,
        spec_id="spec_rep",
        run_id="run_rep",
        outcome_id="out_rep",
        actual_metric=0.85,
        baseline_metric=0.80,
        validity_verdict=ValidityVerdict.PASS,
        evaluator_basis="StandardScientificEvaluator",
        observation_timestamp=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
    )

    # 3. Replay reconstruction of observation
    obs_replayed = service.reconstruct_observation(
        policy_decision=original_record,
        spec_id="spec_rep",
        run_id="run_rep",
        outcome_id="out_rep",
        actual_metric=0.85,
        baseline_metric=0.80,
        validity_verdict=ValidityVerdict.PASS,
        evaluator_basis="StandardScientificEvaluator",
        observation_timestamp=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
    )

    # Fingerprints match bitwise
    assert obs_original.observation_fingerprint() == obs_replayed.observation_fingerprint()
