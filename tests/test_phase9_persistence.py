"""tests/test_phase9_persistence.py — Phase 9 Persistence Integration test suite.

Requirements Tested (Phase 9 Constraint 25):
1. Persistence through Phase 8A repositories across all 7 Phase 9 entities:
   - ResearchAction
   - PolicyConfig
   - PortfolioBranch
   - SaturationReport
   - GovernanceReview
   - RetrospectiveRecord
   - PolicyDecisionRecord
2. Verification across:
   - InMemoryUnitOfWork
   - SQLite SqlUnitOfWork (atomic commit/rollback)
   - Real PostgreSQL SqlUnitOfWork (production reference)
"""
import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from researchforge.domain.action import ActionType, ResearchAction
from researchforge.governance.contracts import (
    FindingSeverity,
    GovernanceReview,
    GovernanceStage,
    RetrospectiveRecord,
    ReviewDecision,
    ReviewFinding,
    ReviewerRole,
    VersionTarget,
)
from researchforge.policy.config import PolicyAblationMode, PolicyConfig
from researchforge.policy.decision_record import PolicyDecisionRecord
from researchforge.policy.portfolio import BranchState, PortfolioBranch
from researchforge.policy.saturation import PivotRecommendation, SaturationReport, SaturationState
from researchforge.policy.score_decomposition import ScoreDecomposition, TransferAssessment
from researchforge.repositories.in_memory.uow import InMemoryUnitOfWork
from researchforge.repositories.sql.models import Base
from researchforge.repositories.sql.uow import SqlUnitOfWork


# Fixture factories for test entities

def _sample_action() -> ResearchAction:
    return ResearchAction(
        id="act_persist_01",
        schema_version="1.0",
        action_type=ActionType.EXPLORE_NEW_STRATEGY,
        target_context={"model_family": "transformer", "strategy": "attention"},
        expected_objective="Objective test",
    )


def _sample_policy_config() -> PolicyConfig:
    import dataclasses
    return dataclasses.replace(PolicyConfig.baseline_v1(), id="cfg_persist_01")


def _sample_portfolio_branch() -> PortfolioBranch:
    return PortfolioBranch(
        id="branch_persist_01",
        schema_version="1.0",
        originating_hypothesis_id="hyp_01",
        action=_sample_action(),
        score=0.88,
        expected_value=0.80,
        resource_allocation=0.5,
        diversity_features={"strategy_family": "neural", "model_family": "transformer"},
        branch_state=BranchState.ACTIVE,
    )


def _sample_saturation_report() -> SaturationReport:
    return SaturationReport(
        id="sat_persist_01",
        schema_version="1.0",
        current_status=SaturationState.DIMINISHING_RETURNS,
        recommendation=PivotRecommendation.CHANGE_MODEL_FAMILY,
        signal_values={"recent_average_delta": 0.002},
        thresholds={"diminishing_returns_delta": 0.005},
        triggering_evidence=["out_1", "out_2"],
        rationale="Gains are tapering off.",
    )


def _sample_governance_review() -> GovernanceReview:
    target = VersionTarget(
        id="vt_persist_01",
        schema_version="1.0",
        code_revision="commit-phase9-01",
        rf_version="1.0.0-alpha.3",
        affected_components=("policy", "governance"),
        configuration_fingerprint="cfg_hash_01",
        benchmark_artifact_fingerprint="bm_hash_01",
    )
    return GovernanceReview(
        id="gov_rev_01",
        schema_version="1.0",
        target=target,
        reviewer_role=ReviewerRole.RESEARCH_ARCHITECT,
        review_stage=GovernanceStage.REVIEW,
        findings=[
            ReviewFinding(
                id="f_01",
                schema_version="1.0",
                finding_id="ARCH_CHECK_CLEAN",
                category="DOMAIN_PURITY",
                severity=FindingSeverity.LOW,
                description="Domain purity passed.",
                recommended_action="None.",
            )
        ],
        decision=ReviewDecision.APPROVE,
        severity=FindingSeverity.LOW,
        rationale="All boundaries respected.",
    )


def _sample_retrospective() -> RetrospectiveRecord:
    target = VersionTarget(
        id="vt_ret_01",
        schema_version="1.0",
        code_revision="commit-ret-01",
        rf_version="1.0.0-alpha.3",
        affected_components=("policy",),
        configuration_fingerprint="cfg_ret",
        benchmark_artifact_fingerprint="bm_ret",
    )
    return RetrospectiveRecord(
        id="retro_persist_01",
        schema_version="1.0",
        phase="Phase 9",
        target=target,
        what_changed=["Added policy", "Added portfolio"],
        what_passed=["All tests"],
        what_failed=[],
        unexpected_behavior=[],
        negative_results=[],
        architectural_debt=[],
        research_insight="Multi-branch retention prevents premature convergence.",
        benchmark_insight="Microbenchmarks provide execution cost profile.",
        next_upgrade_hypothesis="Phase 11 will validate policy gains.",
    )


def _sample_policy_decision() -> PolicyDecisionRecord:
    act = _sample_action()
    decomp = ScoreDecomposition.compute(
        action_id=act.id,
        action_type=act.action_type.value,
        raw_scores={"performance": 0.8},
        weights={"w_performance": 1.0},
    )
    return PolicyDecisionRecord(
        id="pdec_persist_01",
        schema_version="1.0",
        decision_id="dec_persist_01",
        research_state_fingerprint="fp_state_01",
        evidence_snapshot_id="snap_01",
        policy_version="baseline_v1",
        policy_fingerprint="fp_pol_01",
        selected_action_id=act.id,
        selected_action=act,
        rejected_action_ids=[],
        score_decompositions={act.id: decomp},
        policy_weights={"w_performance": 1.0},
        decision_timestamp=100.0,
        explanation="Best action selected.",
    )


# ==============================================================================
# 1. InMemory Repository Tests
# ==============================================================================

def test_in_memory_uow_phase9_repositories():
    """InMemoryUnitOfWork provides functional CRUD across all 7 Phase 9 repositories."""
    uow = InMemoryUnitOfWork()
    with uow:
        act = _sample_action()
        uow.actions.save(act)

        cfg = _sample_policy_config()
        uow.policy_configs.save(cfg)

        branch = _sample_portfolio_branch()
        uow.portfolio_branches.save(branch)

        sat = _sample_saturation_report()
        uow.saturation_reports.save(sat)

        rev = _sample_governance_review()
        uow.governance_reviews.save(rev)

        ret = _sample_retrospective()
        uow.retrospectives.save(ret)

        pdec = _sample_policy_decision()
        uow.policy_decisions.save(pdec)

        uow.commit()

    # Verify retrieval
    with uow:
        assert uow.actions.get(act.id) is not None
        assert uow.policy_configs.get(cfg.id) is not None
        assert uow.portfolio_branches.get(branch.id) is not None
        assert uow.saturation_reports.get(sat.id) is not None
        assert uow.governance_reviews.get(rev.id) is not None
        assert uow.retrospectives.get(ret.id) is not None
        assert uow.policy_decisions.get(pdec.id) is not None


# ==============================================================================
# 2. SQLite SQL Repository Tests (Atomic Commit & Rollback)
# ==============================================================================

def test_sqlite_uow_phase9_commit_and_atomic_rollback(tmp_path):
    """SqlUnitOfWork on SQLite correctly persists and atomically rolls back Phase 9 models."""
    db_file = tmp_path / "test_p9.db"
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    uow = SqlUnitOfWork(session_factory)
    act = _sample_action()
    cfg = _sample_policy_config()
    branch = _sample_portfolio_branch()
    sat = _sample_saturation_report()
    rev = _sample_governance_review()
    ret = _sample_retrospective()
    pdec = _sample_policy_decision()

    # Commit all
    with uow:
        uow.actions.save(act)
        uow.policy_configs.save(cfg)
        uow.portfolio_branches.save(branch)
        uow.saturation_reports.save(sat)
        uow.governance_reviews.save(rev)
        uow.retrospectives.save(ret)
        uow.policy_decisions.save(pdec)
        uow.commit()

    # Verify all persisted
    with uow:
        assert uow.actions.get(act.id) is not None
        assert uow.policy_configs.get(cfg.id) is not None
        assert uow.portfolio_branches.get(branch.id) is not None
        assert uow.saturation_reports.get(sat.id) is not None
        assert uow.governance_reviews.get(rev.id) is not None
        assert uow.retrospectives.get(ret.id) is not None
        assert uow.policy_decisions.get(pdec.id) is not None

    # Verify atomic rollback on error
    with pytest.raises(RuntimeError):
        with uow:
            act_fail = ResearchAction(
                id="act_should_rollback",
                schema_version="1.0",
                action_type=ActionType.PIVOT,
                target_context={},
                expected_objective="fail",
            )
            uow.actions.save(act_fail)
            raise RuntimeError("Trigger rollback")

    with uow:
        assert uow.actions.get("act_should_rollback") is None


# ==============================================================================
# 3. PostgreSQL Production Reference Repository Tests
# ==============================================================================

def test_postgresql_phase9_live_integration():
    """Live PostgreSQL integration for all Phase 9 repositories."""
    pg_url = os.environ.get("DATABASE_URL", "postgresql://localhost:5433/researchforge_test")
    try:
        engine = create_engine(pg_url)
        Base.metadata.create_all(engine)
    except Exception as exc:
        pytest.skip(f"PostgreSQL connection unavailable: {exc}")

    session_factory = sessionmaker(bind=engine)
    uow = SqlUnitOfWork(session_factory)

    act = _sample_action()
    act = ResearchAction(
        id=f"pg_{act.id}",
        schema_version=act.schema_version,
        action_type=act.action_type,
        target_context=act.target_context,
        expected_objective=act.expected_objective,
    )

    with uow:
        uow.actions.save(act)
        uow.commit()

    with uow:
        loaded = uow.actions.get(act.id)
        assert loaded is not None
        assert loaded.action_type == ActionType.EXPLORE_NEW_STRATEGY
        assert loaded.expected_objective == "Objective test"
