"""tests/test_phase9_microbenchmark.py — Deterministic microbenchmarks for Phase 9 components.

Scientific & Operational Principles (Phase 9 Constraints 27, 29):
1. Separately measures component overheads:
   - evidence retrieval
   - policy scoring
   - portfolio selection
   - saturation detection
   - persistence
   - complete policy decision cycle
2. Establishes an empirical baseline, NOT a scientific performance claim.
"""
import time
import pytest
from researchforge.domain.action import ActionType, ResearchAction
from researchforge.domain.evidence import Evidence
from researchforge.domain.outcome import Outcome
from researchforge.domain.state import ResearchState
from researchforge.evidence.snapshot import EvidenceSnapshot
from researchforge.policy.config import PolicyConfig
from researchforge.policy.portfolio import BranchState, PortfolioBranch, ResearchPortfolio
from researchforge.policy.research_policy import ResearchPolicy
from researchforge.policy.saturation import ResearchSaturationDetector
from researchforge.policy.score_decomposition import ScoreDecomposition
from researchforge.repositories.in_memory.uow import InMemoryUnitOfWork


def test_microbenchmark_component_breakdown():
    """Measures separate timing for each core Phase 9 decision component."""
    metrics_report = {}

    # 1. Evidence Retrieval & Snapshot creation
    t0 = time.perf_counter()
    evidence_items = [
        Evidence(
            id=f"ev_bench_{i}",
            schema_version="1.0",
            source="benchmark",
            source_id=f"act_{i % 5}",
            claim_id=f"claim_{i % 3}",
            evidence_type="experimental",
            metadata={"timestamp": 10.0 + i},
        )
        for i in range(50)
    ]
    snapshot = EvidenceSnapshot.create(
        decision_id="dec_bench_01",
        as_of_timestamp=30.0,
        evidence_ids=[e.id for e in evidence_items],
        evidence_items=evidence_items,
    )
    t_ev = time.perf_counter() - t0
    metrics_report["evidence_retrieval_sec"] = t_ev
    assert len(snapshot.evidence_items) == 50

    # 2. Policy Scoring
    policy = ResearchPolicy(config=PolicyConfig.baseline_v1())
    state = ResearchState(id="s_bench", schema_version="1.0")
    candidates = [
        ResearchAction(
            id=f"act_cand_{i}",
            schema_version="1.0",
            action_type=ActionType.EXPLORE_NEW_STRATEGY if i % 2 == 0 else ActionType.EXPLOIT_KNOWN_STRATEGY,
            target_context={"strategy": f"strat_{i % 4}", "model_family": "mlp"},
            expected_objective=f"Objective {i}",
        )
        for i in range(10)
    ]
    t0 = time.perf_counter()
    _, record = policy.evaluate_candidates(state, candidates, snapshot, decision_timestamp=30.0)
    t_policy = time.perf_counter() - t0
    metrics_report["policy_scoring_sec"] = t_policy
    assert len(record.score_decompositions) == 10
    assert record.selected_action_id in [c.id for c in candidates]

    # 3. Portfolio Selection & Diversity
    portfolio = ResearchPortfolio(max_active_branches=5)
    t0 = time.perf_counter()
    for i in range(5):
        b = PortfolioBranch(
            id=f"branch_{i}",
            schema_version="1.0",
            originating_hypothesis_id=f"hyp_{i}",
            action=candidates[i],
            score=0.7 + 0.05 * i,
            expected_value=0.65 + 0.05 * i,
            resource_allocation=0.2,
            diversity_features={"strategy_family": f"family_{i % 3}", "model_family": "mlp"},
            branch_state=BranchState.ACTIVE,
        )
        portfolio.add_branch(b)
    allocations = portfolio.allocate_resources(total_budget=100.0)
    diversity = portfolio.compute_diversity()
    t_port = time.perf_counter() - t0
    metrics_report["portfolio_selection_sec"] = t_port
    assert len(allocations) == 5
    assert diversity["strategy_entropy"] > 0.0

    # 4. Saturation Detection
    detector = ResearchSaturationDetector(min_samples=3)
    outcomes = [
        Outcome(
            id=f"out_bench_{i}",
            schema_version="1.0",
            run_id=f"run_bench_{i}",
            measured_metrics={"metric": 0.80 + 0.005 * i},
        )
        for i in range(8)
    ]
    t0 = time.perf_counter()
    sat_report = detector.evaluate(outcomes)
    t_sat = time.perf_counter() - t0
    metrics_report["saturation_detection_sec"] = t_sat
    assert sat_report is not None

    # 5. Persistence
    uow = InMemoryUnitOfWork()
    t0 = time.perf_counter()
    with uow:
        for c in candidates:
            uow.actions.save(c)
        uow.policy_decisions.save(record)
        uow.saturation_reports.save(sat_report)
        uow.commit()
    t_persist = time.perf_counter() - t0
    metrics_report["persistence_sec"] = t_persist

    # 6. Complete Decision Cycle
    t0 = time.perf_counter()
    with uow:
        d_dec, d_rec = policy.evaluate_candidates(state, candidates, snapshot, decision_timestamp=35.0)
        uow.policy_decisions.save(d_rec)
        uow.commit()
    t_cycle = time.perf_counter() - t0
    metrics_report["complete_cycle_sec"] = t_cycle

    # Print baseline metrics report
    print("\n" + "=" * 60)
    print("PHASE 9 DETERMINISTIC MICROBENCHMARK BASELINE")
    print("=" * 60)
    for k, v in metrics_report.items():
        print(f"  {k:<30}: {v * 1000:8.3f} ms")
    print("=" * 60)

    # Sanity bounds (deterministic execution in reasonable time)
    assert t_ev < 0.5
    assert t_policy < 0.5
    assert t_port < 0.5
    assert t_sat < 0.5
    assert t_persist < 0.5
    assert t_cycle < 0.5
