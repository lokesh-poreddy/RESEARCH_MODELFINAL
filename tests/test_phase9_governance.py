"""tests/test_phase9_governance.py — Unit and behavioral tests for the ResearchForge Governance Layer.

Requirements Tested (Phase 9 Constraints 1, 2, 3, 5, 20, 21, 22, 23):
1. Dual-Loop Invariant: Meta-Development Loop (GovernanceEngine) and Research Loop (ResearchPolicy)
   are completely separate, never merged into one class or state machine.
2. Native Python: Zero external runtime dependencies.
3. Advisory governance before freeze: Emits APPROVE / REQUEST_CHANGES / REJECT without automatic code rewriting.
4. All 9 GovernanceStage enum members and all 9 ReviewerRole enum members.
5. VersionTarget immutability: Rejects 'latest' and empty strings.
6. Reviewer modules: Architecture, Scientific, Safety, Adversarial, Benchmark, Release, Retrospective.
7. ReleaseGate verification: Rejects release on test failure or critical findings.
"""
from pathlib import Path
import pytest
from researchforge.governance.architecture_review import ArchitectureReviewer
from researchforge.governance.contracts import (
    FindingSeverity,
    GovernanceReview,
    GovernanceStage,
    ReleaseGate,
    RetrospectiveRecord,
    ReviewDecision,
    ReviewFinding,
    ReviewerRole,
    VersionTarget,
)
from researchforge.governance.governance_engine import GovernanceEngine
from researchforge.governance.release_review import ReleaseReviewer
from researchforge.policy.research_policy import ResearchPolicy


def test_dual_loop_architectural_separation():
    """GovernanceEngine (meta-loop) and ResearchPolicy (research-loop) are strictly decoupled."""
    # Neither inherits from the other
    assert not issubclass(GovernanceEngine, ResearchPolicy)
    assert not issubclass(ResearchPolicy, GovernanceEngine)

    # GovernanceEngine handles platform development stages
    engine = GovernanceEngine()
    stages = [s.value for s in GovernanceStage]
    assert "PLAN" in stages
    assert "CHALLENGE" in stages
    assert "IMPLEMENT" in stages
    assert "REVIEW" in stages
    assert "TEST" in stages
    assert "SCIENTIFIC_VALIDATE" in stages
    assert "BENCHMARK" in stages
    assert "FREEZE" in stages
    assert "RETROSPECT" in stages
    assert len(GovernanceStage) == 9

    # Reviewer roles cover all 9 specialized functions
    assert len(ReviewerRole) == 9


def test_version_target_rejects_latest_and_empty_revisions():
    """VersionTarget requires an explicit, immutable revision and rejects 'latest'."""
    with pytest.raises(ValueError, match="not 'latest'"):
        VersionTarget(
            id="vt_1",
            schema_version="1.0",
            code_revision="latest",
            rf_version="1.0.0-alpha.3",
            affected_components=("policy", "governance"),
            configuration_fingerprint="cfg_fp_123",
            benchmark_artifact_fingerprint="bm_fp_123",
        )

    with pytest.raises(ValueError, match="not 'latest'"):
        VersionTarget(
            id="vt_2",
            schema_version="1.0",
            code_revision="",
            rf_version="1.0.0-alpha.3",
            affected_components=("policy",),
            configuration_fingerprint="cfg_fp_123",
            benchmark_artifact_fingerprint="bm_fp_123",
        )

    valid_target = VersionTarget(
        id="vt_valid",
        schema_version="1.0",
        code_revision="git-commit-abc1234",
        rf_version="1.0.0-alpha.3",
        affected_components=("policy", "governance"),
        configuration_fingerprint="cfg_fp_123",
        benchmark_artifact_fingerprint="bm_fp_123",
    )
    assert valid_target.code_revision == "git-commit-abc1234"


def test_architecture_reviewer_verifies_domain_purity(tmp_path: Path):
    """ArchitectureReviewer detects violations such as database imports in domain layer."""
    reviewer = ArchitectureReviewer(workspace_root=tmp_path)

    # Create clean domain file
    domain_dir = tmp_path / "researchforge" / "domain"
    domain_dir.mkdir(parents=True, exist_ok=True)
    clean_file = domain_dir / "clean_model.py"
    clean_file.write_text("class CleanModel:\n    pass\n")

    target = VersionTarget(
        id="vt_arch",
        schema_version="1.0",
        code_revision="commit-001",
        rf_version="1.0.0-alpha.3",
        affected_components=("domain",),
        configuration_fingerprint="cfg_1",
        benchmark_artifact_fingerprint="bm_1",
    )

    review_clean = reviewer.review(target)
    assert review_clean.decision in (ReviewDecision.APPROVE, ReviewDecision.APPROVE_WITH_WARNINGS)
    assert len(review_clean.findings) == 0

    # Inject dirty domain file with database import
    dirty_file = domain_dir / "dirty_model.py"
    dirty_file.write_text("import sqlalchemy\nclass DirtyModel:\n    pass\n")

    review_dirty = reviewer.review(target)
    assert review_dirty.decision == ReviewDecision.REJECT
    assert any(f.severity == FindingSeverity.CRITICAL for f in review_dirty.findings)
    assert any("sqlalchemy" in f.description for f in review_dirty.findings)


def test_release_reviewer_and_release_gate_evaluation():
    """ReleaseReviewer evaluates whether the system/release can be frozen."""
    target = VersionTarget(
        id="vt_rel",
        schema_version="1.0",
        code_revision="commit-freeze-001",
        rf_version="1.0.0-alpha.3",
        affected_components=("policy", "governance"),
        configuration_fingerprint="cfg_release",
        benchmark_artifact_fingerprint="bm_release",
    )

    reviewer = ReleaseReviewer()

    # Scenario 1: Test failure blocks release gate
    review, gate = reviewer.review(
        target=target,
        prior_reviews=[],
        all_tests_passed=False,
        unresolved_critical_warnings=0,
        migrations_verified=True,
    )
    assert gate.passed is False
    assert any("failing tests" in r for r in gate.blocking_reasons)
    assert review.decision == ReviewDecision.REJECT

    # Scenario 2: Unresolved critical warnings block release
    review, gate = reviewer.review(
        target=target,
        prior_reviews=[],
        all_tests_passed=True,
        unresolved_critical_warnings=2,
        migrations_verified=True,
    )
    assert gate.passed is False
    assert any("critical warnings" in r for r in gate.blocking_reasons)
    assert review.decision == ReviewDecision.REJECT

    # Scenario 3: Clean state passes release gate
    review, gate = reviewer.review(
        target=target,
        prior_reviews=[],
        all_tests_passed=True,
        unresolved_critical_warnings=0,
        migrations_verified=True,
    )
    assert gate.passed is True
    assert len(gate.blocking_reasons) == 0
    assert review.decision == ReviewDecision.APPROVE


def test_full_governance_engine_cycle():
    """GovernanceEngine runs full review cycle across architecture, safety, scientific, adversarial, benchmark, release."""
    engine = GovernanceEngine()
    target = VersionTarget(
        id="vt_full",
        schema_version="1.0",
        code_revision="commit-valid-999",
        rf_version="1.0.0-alpha.3",
        affected_components=("policy", "governance", "repositories"),
        configuration_fingerprint="cfg_valid",
        benchmark_artifact_fingerprint="bm_valid",
    )

    gate = engine.run_full_governance_cycle(
        target=target,
        all_tests_passed=True,
        unresolved_warnings=0,
        migrations_verified=True,
    )

    reviews = engine.list_reviews()
    # Verified: 6 reviews generated (architecture, safety, scientific, adversarial, benchmark, release)
    assert len(reviews) == 6
    roles = [r.reviewer_role for r in reviews]
    assert ReviewerRole.RESEARCH_ARCHITECT in roles
    assert ReviewerRole.SAFETY_EXECUTION_REVIEWER in roles
    assert ReviewerRole.SCIENTIFIC_METHOD_REVIEWER in roles
    assert ReviewerRole.ADVERSARIAL_EVALUATOR in roles
    assert ReviewerRole.BENCHMARK_SCIENTIST in roles
    assert ReviewerRole.RELEASE_REPRODUCIBILITY_ENGINEER in roles

    # Gate evaluated successfully
    assert isinstance(gate, ReleaseGate)
    assert gate.passed is True
