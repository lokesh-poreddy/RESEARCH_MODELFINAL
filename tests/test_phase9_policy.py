"""tests/test_phase9_policy.py — Phase 9 ResearchPolicy, Action & Evaluators Test Suite.

Verifies:
1. Canonical ResearchAction construction, schema validation, and ActionType enum.
2. PolicyConfig explicit baseline heuristic weights ('baseline_v1' / 'heuristic_prior').
3. Transparent ScoreDecomposition with both raw scores and weighted contributions visible.
4. Human-readable explanation generation from components ('Why was this action selected?').
5. Policy ablation modes (RANDOM, STATIC_BASELINE, UCB, MEMORY_CONDITIONED, ADAPTIVE_POLICY).
6. Failure-aware scoring: repeated contextual failure increases risk without universal prohibition.
7. Transfer categorization (POSITIVE_TRANSFER, NEGATIVE_TRANSFER, etc.).
8. Independent evaluator boundaries (ResearchPolicy vs ScientificEvaluator vs BenchmarkEvaluator).
9. PolicyGate update proposals, deterministic fixture validation, and version progression.
"""
from __future__ import annotations

import random
import time
import pytest

from researchforge.domain.action import ActionType, ResearchAction
from researchforge.domain.experiment import ExecutionStatus, ExperimentRun, ExperimentSpec
from researchforge.domain.outcome import Outcome
from researchforge.domain.state import ResearchState
from researchforge.domain.validity import ValidityVerdict
from researchforge.evidence.snapshot import EvidenceSnapshot
from researchforge.memory.adaptive_trajectory import (
    AdaptiveTrajectoryMemory,
    AdaptiveTrajectoryRecord,
    CapacityBucket,
    Stage,
)
from researchforge.policy.config import PolicyAblationMode, PolicyConfig
from researchforge.policy.decision_record import PolicyDecisionRecord
from researchforge.policy.evaluators import (
    BenchmarkEvaluator,
    ScientificEvaluator,
    StandardBenchmarkEvaluator,
    StandardScientificEvaluator,
)
from researchforge.policy.gating import PolicyGate, PolicyUpdateProposal
from researchforge.policy.research_policy import ResearchPolicy
from researchforge.policy.score_decomposition import ScoreDecomposition, TransferAssessment


# ==============================================================================
# 1. ResearchAction & ActionType
# ==============================================================================

def test_research_action_creation_and_fingerprint():
    """ResearchAction is canonical, frozen, and content-addressed."""
    action = ResearchAction(
        id="act_01",
        schema_version="1.0",
        action_type=ActionType.EXPLORE_NEW_STRATEGY,
        target_context={"problem_id": "prob_1", "strategy": "width_scale", "model_family": "mlp"},
        expected_objective="Discover if width scaling improves generalization",
        parameters={"multiplier": 2.0, "cost": 0.3},
    )
    assert action.id == "act_01"
    assert action.action_type == ActionType.EXPLORE_NEW_STRATEGY
    assert action.target_context["strategy"] == "width_scale"
    assert action.fingerprint() != ""

    # Serialization roundtrip
    d = action.to_dict()
    reconstructed = ResearchAction.from_dict(d)
    assert reconstructed.fingerprint() == action.fingerprint()
    assert reconstructed.action_type == ActionType.EXPLORE_NEW_STRATEGY


# ==============================================================================
# 2. PolicyConfig: Explicit Baseline Prior
# ==============================================================================

def test_policy_config_heuristic_baseline_prior_labeling():
    """PolicyConfig default weights are explicitly labeled as heuristic baseline prior, not optimal."""
    cfg = PolicyConfig.baseline_v1()
    assert cfg.policy_version == "baseline_v1"
    assert cfg.config_label == "heuristic_prior"
    assert cfg.w_performance == 1.0
    assert cfg.w_failure_risk == 1.2
    assert cfg.w_cost == 0.3

    # Ensure all weights are visible in dictionary
    w = cfg.weights_dict()
    assert len(w) == 8
    assert "w_performance" in w
    assert "w_failure_risk" in w
    assert "w_transfer" in w


# ==============================================================================
# 3. Transparent ScoreDecomposition & Explanation
# ==============================================================================

def test_score_decomposition_transparency_and_explanation():
    """ScoreDecomposition reveals raw values, weighted values, total utility, and auditable explanation."""
    raw = {
        "performance": 0.8,
        "information_gain": 0.6,
        "novelty": 0.5,
        "evidence": 0.4,
        "transfer": 0.7,
        "failure_risk": 0.1,
        "redundancy": 0.0,
        "cost": 0.2,
    }
    cfg = PolicyConfig.baseline_v1()
    weights = cfg.weights_dict()

    decomp = ScoreDecomposition.compute(
        action_id="act_test",
        action_type="EXPLORE_NEW_STRATEGY",
        raw_scores=raw,
        weights=weights,
        transfer_assessment=TransferAssessment.POSITIVE_TRANSFER,
    )

    # Positive sum: 1.0*0.8 + 0.8*0.6 + 0.5*0.5 + 0.7*0.4 + 0.6*0.7 = 0.8 + 0.48 + 0.25 + 0.28 + 0.42 = 2.23
    # Negative sum: 1.2*0.1 + 0.8*0.0 + 0.3*0.2 = 0.12 + 0.0 + 0.06 = 0.18
    # Utility = 2.23 - 0.18 = 2.05
    expected_utility = 2.05
    assert abs(decomp.total_utility - expected_utility) < 1e-4

    # Check weighted contributions match
    assert abs(decomp.w_performance - 0.8) < 1e-4
    assert abs(decomp.w_failure_risk - 0.12) < 1e-4
    assert abs(decomp.w_cost - 0.06) < 1e-4

    # Explanation is deterministic and generated directly from components
    assert "act_test" in decomp.explanation
    assert "utility=2.0500" in decomp.explanation
    assert "POSITIVE_TRANSFER" in decomp.explanation


# ==============================================================================
# 4. ResearchPolicy Evaluation & Decision Record
# ==============================================================================

def test_research_policy_evaluation_produces_auditable_records():
    """ResearchPolicy produces both domain Decision and detailed PolicyDecisionRecord."""
    policy = ResearchPolicy(config=PolicyConfig.baseline_v1())
    state = ResearchState(id="s_eval_01", schema_version="1.0", problem_id="prob_1")

    actions = [
        ResearchAction(id="act_exp", schema_version="1.0", action_type=ActionType.EXPLORE_NEW_STRATEGY, target_context={"strategy": "deepen"}, expected_objective="Explore depth"),
        ResearchAction(id="act_rep", schema_version="1.0", action_type=ActionType.REPLICATE, target_context={"strategy": "replicate"}, expected_objective="Replicate"),
    ]
    snapshot = EvidenceSnapshot.create(decision_id="dec_01", as_of_timestamp=100.0, evidence_ids=[])

    domain_dec, policy_rec = policy.evaluate_candidates(
        state=state,
        candidate_actions=actions,
        snapshot=snapshot,
        decision_timestamp=100.0,
    )

    # Domain decision checks
    assert domain_dec.research_state_fingerprint == state.fingerprint()
    assert domain_dec.decision_reason != ""
    assert domain_dec.policy_version == "baseline_v1"

    # Detailed policy decision record checks
    assert policy_rec.research_state_fingerprint == state.fingerprint()
    assert policy_rec.evidence_snapshot_id == snapshot.id
    assert policy_rec.selected_action_id in ["act_exp", "act_rep"]
    assert len(policy_rec.rejected_action_ids) == 1
    assert len(policy_rec.score_decompositions) == 2
    assert policy_rec.credit_assignment_meta is not None
    assert policy_rec.credit_assignment_meta["prior_state_id"] == "s_eval_01"

    # Deterministic decision fingerprint exists
    fp = policy_rec.decision_fingerprint()
    assert len(fp) == 64


# ==============================================================================
# 5. Failure-Aware Policy Scoring
# ==============================================================================

def test_failure_awareness_penalizes_repeated_contextual_failures_without_blanket_ban():
    """Repeated failure in same context increases risk; one isolated failure does not universally ban."""
    mem = AdaptiveTrajectoryMemory(min_context_samples=2)

    # Context: problem=p1, model=mlp, bucket=MEDIUM, strategy=scaling
    # Add 1 failure
    mem.store(
        AdaptiveTrajectoryRecord(
            id="traj_f1",
            generation=1,
            stage="early",
            problem_context="p1",
            parent_model_type="mlp",
            parent_capacity_bucket="medium",
            strategy="scaling",
            child_model_type="mlp",
            child_capacity_bucket="medium",
            metric=0.2,
            success=False,
            failure="OOM_ERROR",
        )
    )

    policy = ResearchPolicy(config=PolicyConfig.baseline_v1(), memory=mem)
    state = ResearchState(id="s_faware", schema_version="1.0", problem_id="p1")
    act_same_context = ResearchAction(
        id="act_same",
        schema_version="1.0",
        action_type=ActionType.EXPLOIT_KNOWN_STRATEGY,
        target_context={"problem_id": "p1", "model_family": "mlp", "capacity_bucket": "medium", "strategy": "scaling"},
        expected_objective="Try scaling again",
    )
    act_diff_context = ResearchAction(
        id="act_diff",
        schema_version="1.0",
        action_type=ActionType.EXPLORE_NEW_STRATEGY,
        target_context={"problem_id": "p2", "model_family": "cnn", "capacity_bucket": "low", "strategy": "conv_branch"},
        expected_objective="Try cnn in p2",
    )
    snap = EvidenceSnapshot.create(decision_id="d_f", as_of_timestamp=100.0, evidence_ids=[])

    _, rec1 = policy.evaluate_candidates(state, [act_same_context, act_diff_context], snap)
    risk_same_1 = rec1.score_decompositions["act_same"].failure_risk
    risk_diff_1 = rec1.score_decompositions["act_diff"].failure_risk

    # 1 failure increases risk in same context compared to unfailed context, but does NOT ban (risk < 1.0)
    assert risk_same_1 > risk_diff_1
    assert risk_same_1 < 0.9

    # Add 2 more failures to the same context (repeated contextual failures)
    for i in range(2, 4):
        mem.store(
            AdaptiveTrajectoryRecord(
                id=f"traj_f{i}",
                generation=i,
                stage="early",
                problem_context="p1",
                parent_model_type="mlp",
                parent_capacity_bucket="medium",
                strategy="scaling",
                child_model_type="mlp",
                child_capacity_bucket="medium",
                metric=0.1,
                success=False,
                failure="OOM_ERROR",
            )
        )

    _, rec2 = policy.evaluate_candidates(state, [act_same_context, act_diff_context], snap)
    risk_same_3 = rec2.score_decompositions["act_same"].failure_risk
    assert risk_same_3 > risk_same_1  # Risk scales with repeated contextual failures!


# ==============================================================================
# 6. Policy Ablation Modes
# ==============================================================================

def test_policy_ablation_modes_execute_consistently():
    """All 5 ablation modes evaluate candidates across identical tasks and state."""
    state = ResearchState(id="s_abl", schema_version="1.0")
    actions = [
        ResearchAction(id="a1", schema_version="1.0", action_type=ActionType.EXPLORE_NEW_STRATEGY, target_context={"strategy": "s1"}, expected_objective="s1"),
        ResearchAction(id="a2", schema_version="1.0", action_type=ActionType.EXPLOIT_KNOWN_STRATEGY, target_context={"strategy": "s2"}, expected_objective="s2"),
    ]
    snap = EvidenceSnapshot.create(decision_id="d_abl", as_of_timestamp=100.0, evidence_ids=[])

    for mode in [
        PolicyAblationMode.RANDOM,
        PolicyAblationMode.STATIC_BASELINE,
        PolicyAblationMode.UCB,
        PolicyAblationMode.MEMORY_CONDITIONED,
        PolicyAblationMode.ADAPTIVE_POLICY,
    ]:
        cfg = PolicyConfig.baseline_v1(ablation_mode=mode)
        p = ResearchPolicy(config=cfg, rng=random.Random(123))
        dec, rec = p.evaluate_candidates(state, actions, snap, decision_timestamp=100.0)
        assert dec is not None
        assert rec.selected_action_id in ["a1", "a2"]


# ==============================================================================
# 7. Independent Evaluator Boundary
# ==============================================================================

def test_independent_evaluator_boundary():
    """ScientificEvaluator and BenchmarkEvaluator operate independently of policy internal scores."""
    sci_eval = StandardScientificEvaluator()
    bench_eval = StandardBenchmarkEvaluator()

    spec = ExperimentSpec(id="sp_1", schema_version="1.0", decision_id="dec_1")
    run_success = ExperimentRun(id="run_1", schema_version="1.0", experiment_spec_id="sp_1", status=ExecutionStatus.SUCCESS)
    out_valid = Outcome(id="out_1", schema_version="1.0", run_id="run_1", measured_metrics={"metric": 0.88})

    # Scientific evaluation of outcome validity
    validity = sci_eval.evaluate_outcome(spec, run_success, out_valid)
    assert validity.verdict == ValidityVerdict.PASS

    # Benchmark comparative evaluation
    out_base = Outcome(id="out_0", schema_version="1.0", run_id="run_0", measured_metrics={"metric": 0.80})
    bench_res = bench_eval.evaluate_benchmark(
        benchmark_id="bench_01",
        results=[out_valid],
        baseline_results=[out_base],
    )
    assert bench_res["independent_evaluation_verified"] is True
    assert bench_res["empirical_gain"] == pytest.approx(0.08, 1e-4)


# ==============================================================================
# 8. PolicyGate: Gated Policy Version Progression
# ==============================================================================

def test_policy_gate_update_validation():
    """PolicyGate validates schema, rejects duplicate versions, and gates activation on governance approval."""
    active_policy = ResearchPolicy(config=PolicyConfig.baseline_v1())
    gate = PolicyGate(active_policy)

    # 1. Proposal with duplicate version -> Rejected
    prop_dup = PolicyUpdateProposal(
        id="prop_01",
        schema_version="1.0",
        old_policy_version="baseline_v1",
        proposed_policy_version="baseline_v1",
        reason="Testing duplicate",
        proposed_config=PolicyConfig.baseline_v1(),
    )
    res_dup = gate.evaluate_proposal(prop_dup)
    assert res_dup.approved is False
    assert "version_monotonicity" in res_dup.validation_checks_failed

    # 2. Proposal with new version but governance rejected -> Rejected
    new_cfg = PolicyConfig(
        id="policy_cfg_v2",
        schema_version="1.0",
        policy_version="policy_v2",
        w_performance=1.2,
    )
    prop_v2 = PolicyUpdateProposal(
        id="prop_02",
        schema_version="1.0",
        old_policy_version="baseline_v1",
        proposed_policy_version="policy_v2",
        reason="Increased performance weighting",
        proposed_config=new_cfg,
    )
    res_gov_blocked = gate.evaluate_proposal(prop_v2, governance_approved=False)
    assert res_gov_blocked.approved is False
    assert "governance_review" in res_gov_blocked.validation_checks_failed

    # 3. Valid proposal with governance approval -> Approved and promoted
    res_ok = gate.evaluate_proposal(prop_v2, governance_approved=True)
    assert res_ok.approved is True
    assert gate.active_policy.policy_version == "policy_v2"
