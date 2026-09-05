"""tests/test_phase9_portfolio.py — Unit and property tests for ResearchPortfolio.

Requirements Tested (Phase 9 Constraints 16 & 17):
1. Multi-branch tracking with full metadata (hypothesis, action, score, allocation, status, diversity features, provenance).
2. Negative branch retention: failing or losing branches are preserved permanently, never deleted.
3. Exposed diversity calculation based on actual research properties (strategy family, model family) via Shannon entropy.
4. Budget allocation across competing branches without complete collapse to single branch.
"""
import pytest
from researchforge.domain.action import ActionType, ResearchAction
from researchforge.policy.portfolio import BranchState, PortfolioBranch, ResearchPortfolio


def test_multibranch_tracking_and_retention():
    """Portfolio maintains multiple active branches and retains metadata."""
    portfolio = ResearchPortfolio(max_active_branches=4)

    act1 = ResearchAction(
        id="act_p1",
        schema_version="1.0",
        action_type=ActionType.EXPLORE_NEW_STRATEGY,
        target_context={"strategy": "transformer", "model_family": "attention"},
        expected_objective="Explore attention",
    )
    act2 = ResearchAction(
        id="act_p2",
        schema_version="1.0",
        action_type=ActionType.EXPLOIT_KNOWN_STRATEGY,
        target_context={"strategy": "gradient_boosting", "model_family": "tree"},
        expected_objective="Exploit trees",
    )

    b1 = PortfolioBranch(
        id="branch_1",
        schema_version="1.0",
        originating_hypothesis_id="hyp_01",
        action=act1,
        score=0.85,
        expected_value=0.80,
        resource_allocation=0.6,
        diversity_features={"strategy_family": "neural", "model_family": "transformer"},
        branch_state=BranchState.ACTIVE,
        provenance_id="prov_01",
    )
    b2 = PortfolioBranch(
        id="branch_2",
        schema_version="1.0",
        originating_hypothesis_id="hyp_02",
        action=act2,
        score=0.72,
        expected_value=0.70,
        resource_allocation=0.4,
        diversity_features={"strategy_family": "ensemble", "model_family": "tree"},
        branch_state=BranchState.ACTIVE,
        provenance_id="prov_02",
    )

    portfolio.add_branch(b1)
    portfolio.add_branch(b2)

    assert len(portfolio.list_branches()) == 2
    assert len(portfolio.list_branches(BranchState.ACTIVE)) == 2
    retrieved = portfolio.get_branch("branch_1")
    assert retrieved is not None
    assert retrieved.originating_hypothesis_id == "hyp_01"
    assert retrieved.provenance_id == "prov_01"


def test_negative_branches_are_preserved_not_discarded():
    """Losing or failing branches must not be silently deleted."""
    portfolio = ResearchPortfolio(max_active_branches=3)

    act = ResearchAction(
        id="act_fail",
        schema_version="1.0",
        action_type=ActionType.RETRY_AFTER_FAILURE,
        target_context={"strategy": "dense_layer", "model_family": "mlp"},
        expected_objective="Dense retry",
    )
    b = PortfolioBranch(
        id="branch_fail",
        schema_version="1.0",
        originating_hypothesis_id="hyp_fail",
        action=act,
        score=0.3,
        expected_value=0.2,
        resource_allocation=0.2,
        diversity_features={"strategy_family": "feedforward", "model_family": "mlp"},
        branch_state=BranchState.ACTIVE,
    )
    portfolio.add_branch(b)

    # Mark failed
    portfolio.mark_branch_failed("branch_fail", "divergence_nan")

    # Verify branch is STILL in portfolio
    assert portfolio.get_branch("branch_fail") is not None
    failed_branches = portfolio.list_branches(BranchState.FAILED)
    assert len(failed_branches) == 1
    assert failed_branches[0].id == "branch_fail"
    assert "divergence_nan" in failed_branches[0].failure_history
    assert failed_branches[0].resource_allocation == 0.0

    # Total branches remains 1, not 0 (no deletion!)
    assert len(portfolio.list_branches()) == 1


def test_exposed_diversity_calculation():
    """Diversity calculation is exposed and computed via Shannon entropy over research properties."""
    portfolio = ResearchPortfolio()

    # 3 branches with distinct strategies and models
    b1 = PortfolioBranch(
        id="b1",
        schema_version="1.0",
        originating_hypothesis_id="h1",
        action=ResearchAction(id="a1", schema_version="1.0", action_type=ActionType.EXPLORE_NEW_STRATEGY, target_context={}, expected_objective="obj1"),
        score=0.8,
        expected_value=0.7,
        resource_allocation=0.33,
        diversity_features={"strategy_family": "neural", "model_family": "transformer"},
        branch_state=BranchState.ACTIVE,
    )
    b2 = PortfolioBranch(
        id="b2",
        schema_version="1.0",
        originating_hypothesis_id="h2",
        action=ResearchAction(id="a2", schema_version="1.0", action_type=ActionType.EXPLORE_NEW_STRATEGY, target_context={}, expected_objective="obj2"),
        score=0.75,
        expected_value=0.65,
        resource_allocation=0.33,
        diversity_features={"strategy_family": "kernel", "model_family": "svm"},
        branch_state=BranchState.ACTIVE,
    )
    b3 = PortfolioBranch(
        id="b3",
        schema_version="1.0",
        originating_hypothesis_id="h3",
        action=ResearchAction(id="a3", schema_version="1.0", action_type=ActionType.EXPLORE_NEW_STRATEGY, target_context={}, expected_objective="obj3"),
        score=0.7,
        expected_value=0.6,
        resource_allocation=0.34,
        diversity_features={"strategy_family": "ensemble", "model_family": "forest"},
        branch_state=BranchState.ACTIVE,
    )

    portfolio.add_branch(b1)
    portfolio.add_branch(b2)
    portfolio.add_branch(b3)

    metrics = portfolio.compute_diversity()
    # 3 distinct families: Shannon entropy is log2(3) ≈ 1.585
    assert metrics["strategy_entropy"] == pytest.approx(1.585, 1e-2)
    assert metrics["model_family_entropy"] == pytest.approx(1.585, 1e-2)
    assert metrics["active_branches"] == 3
    assert metrics["total_branches"] == 3


def test_resource_allocation_prevents_complete_starvation():
    """Allocation distributes total budget proportionally without zeroing out competing branches."""
    portfolio = ResearchPortfolio()
    b1 = PortfolioBranch(
        id="b1",
        schema_version="1.0",
        originating_hypothesis_id="h1",
        action=ResearchAction(id="a1", schema_version="1.0", action_type=ActionType.EXPLOIT_KNOWN_STRATEGY, target_context={}, expected_objective="obj1"),
        score=0.9,
        expected_value=0.85,
        resource_allocation=0.0,
        diversity_features={"strategy_family": "neural", "model_family": "mlp"},
    )
    b2 = PortfolioBranch(
        id="b2",
        schema_version="1.0",
        originating_hypothesis_id="h2",
        action=ResearchAction(id="a2", schema_version="1.0", action_type=ActionType.EXPLORE_NEW_STRATEGY, target_context={}, expected_objective="obj2"),
        score=0.6,
        expected_value=0.55,
        resource_allocation=0.0,
        diversity_features={"strategy_family": "ensemble", "model_family": "tree"},
    )
    portfolio.add_branch(b1)
    portfolio.add_branch(b2)

    allocations = portfolio.allocate_resources(total_budget=100.0)
    assert len(allocations) == 2
    assert allocations["b1"] > allocations["b2"]
    assert allocations["b2"] > 0.0  # b2 is not starved!
    assert sum(allocations.values()) == pytest.approx(100.0, 1e-3)
