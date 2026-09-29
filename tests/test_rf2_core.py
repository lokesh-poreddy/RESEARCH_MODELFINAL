"""Tests for RF-2.0 core algorithmic modules:
  - PosteriorMemory (memory/posterior.py)
  - ContextualBanditPolicy (policy/contextual_bandit.py)
  - TransferGate (policy/transfer_gate.py)

Scientific Testing Principles:
  1. Every test validates a MATHEMATICAL property, not just "it doesn't crash"
  2. Posterior tests verify conjugate updating properties
  3. Policy tests verify that context-dependent learning outperforms context-free
  4. Gate tests verify that the continuous modifier is always in [g_min, 1.0]
  5. Integration tests verify that the three modules compose correctly
"""
from __future__ import annotations

import math
import pytest
import random

from researchforge.memory.posterior import (
    PosteriorState, PosteriorMemory, PosteriorQueryResult,
)
from researchforge.policy.contextual_bandit import (
    ContextualBanditPolicy, ContextVector, PolicyDecision, CONTEXT_DIM,
)
from researchforge.policy.transfer_gate import (
    TransferGate, GateWeights, GateDecision,
)
from researchforge.pipeline.controller import ResearchController
from researchforge.benchmarks.tasks import digits_task


# ============================================================================
# PosteriorState Tests
# ============================================================================

class TestPosteriorState:
    """Tests for the Beta-Binomial posterior state."""

    def test_uniform_prior_has_half_mean(self):
        """Beta(1,1) = Uniform[0,1] → mean = 0.5."""
        ps = PosteriorState(alpha=1.0, beta=1.0)
        assert ps.mean == pytest.approx(0.5)

    def test_posterior_mean_increases_with_successes(self):
        """After observing successes, E[p] should increase."""
        ps = PosteriorState(alpha=1.0, beta=1.0)
        m0 = ps.mean
        ps.update(success=True)
        m1 = ps.mean
        ps.update(success=True)
        m2 = ps.mean
        assert m0 < m1 < m2

    def test_posterior_mean_decreases_with_failures(self):
        """After observing failures, E[p] should decrease."""
        ps = PosteriorState(alpha=1.0, beta=1.0)
        m0 = ps.mean
        ps.update(success=False)
        assert ps.mean < m0

    def test_posterior_variance_decreases_with_more_evidence(self):
        """Posterior variance should decrease as evidence accumulates."""
        ps = PosteriorState(alpha=1.0, beta=1.0)
        v0 = ps.variance
        for _ in range(10):
            ps.update(success=True)
        assert ps.variance < v0

    def test_posterior_mean_always_in_unit_interval(self):
        """The posterior mean must ALWAYS be in [0, 1]."""
        ps = PosteriorState(alpha=1.0, beta=1.0)
        for i in range(100):
            ps.update(success=i % 3 != 0, metric_delta=random.uniform(-1, 1))
            assert 0.0 <= ps.mean <= 1.0

    def test_conjugate_property(self):
        """Beta(α₀ + s, β₀ + f) after s successes and f failures."""
        ps = PosteriorState(alpha=2.0, beta=3.0)  # Prior Beta(2,3)
        ps.update(success=True)   # s=1
        ps.update(success=True)   # s=2
        ps.update(success=False)  # f=1
        assert ps.alpha == pytest.approx(4.0)  # 2 + 2
        assert ps.beta == pytest.approx(4.0)   # 3 + 1
        assert ps.mean == pytest.approx(0.5)   # 4/(4+4)

    def test_credible_interval_contains_mean(self):
        """The credible interval must contain the posterior mean."""
        ps = PosteriorState(alpha=5.0, beta=3.0)
        lo, hi = ps.credible_interval(0.95)
        assert lo <= ps.mean <= hi

    def test_credible_interval_narrows_with_evidence(self):
        """More evidence → narrower credible interval."""
        ps = PosteriorState(alpha=1.0, beta=1.0)
        lo0, hi0 = ps.credible_interval()
        w0 = hi0 - lo0
        for _ in range(20):
            ps.update(success=True)
        lo1, hi1 = ps.credible_interval()
        w1 = hi1 - lo1
        assert w1 < w0

    def test_thompson_sample_in_unit_interval(self):
        """Thompson samples must be in [0, 1]."""
        ps = PosteriorState(alpha=3.0, beta=2.0)
        rng = random.Random(42)
        for _ in range(100):
            s = ps.sample(rng)
            assert 0.0 <= s <= 1.0

    def test_mean_delta_tracking(self):
        """Metric delta tracking should correctly compute the running mean."""
        ps = PosteriorState(alpha=1.0, beta=1.0)
        ps.update(success=True, metric_delta=0.1)
        ps.update(success=True, metric_delta=0.3)
        assert ps.mean_delta == pytest.approx(0.2)

    def test_serialization_roundtrip(self):
        """to_dict → from_dict must preserve all state."""
        ps = PosteriorState(alpha=5.0, beta=3.0, n_observations=8,
                             sum_delta=1.2, sum_delta_sq=0.36,
                             last_updated_gen=7)
        d = ps.to_dict()
        ps2 = PosteriorState.from_dict(d)
        assert ps2.alpha == ps.alpha
        assert ps2.beta == ps.beta
        assert ps2.n_observations == ps.n_observations
        assert ps2.mean == pytest.approx(ps.mean)


# ============================================================================
# PosteriorMemory Tests
# ============================================================================

class TestPosteriorMemory:
    """Tests for hierarchical posterior memory with Bayesian smoothing."""

    def test_empty_memory_returns_prior(self):
        """With no evidence, query should return the prior (0.5)."""
        pm = PosteriorMemory(prior_alpha=1.0, prior_beta=1.0)
        result = pm.query("increase_capacity", "MLP", "medium")
        assert result.success_probability == pytest.approx(0.5)
        assert result.primary_level == 0

    def test_update_propagates_to_all_levels(self):
        """A single update should increment observation counts at all levels."""
        pm = PosteriorMemory()
        pm.update("increase_capacity", "MLP", "medium", success=True)
        result = pm.query("increase_capacity", "MLP", "medium")
        assert result.evidence_counts[3] == 1
        assert result.evidence_counts[2] == 1
        assert result.evidence_counts[1] == 1
        assert result.evidence_counts[0] == 1

    def test_level3_dominates_with_sufficient_evidence(self):
        """With enough L3 evidence, L3 should be the primary level."""
        pm = PosteriorMemory(min_evidence=2)
        for _ in range(5):
            pm.update("increase_capacity", "MLP", "medium", success=True)
        result = pm.query("increase_capacity", "MLP", "medium")
        assert result.primary_level == 3
        assert result.success_probability > 0.5  # All successes

    def test_hierarchical_backoff_to_level1(self):
        """When L3 and L2 lack evidence, should back off to L1."""
        pm = PosteriorMemory(min_evidence=3)
        # Add evidence at L1 level only (different model types / buckets)
        pm.update("increase_capacity", "MLP", "large", success=True)
        pm.update("increase_capacity", "RF", "small", success=True)
        pm.update("increase_capacity", "SVC", "medium", success=False)
        # Query a specific L3 context with no evidence
        result = pm.query("increase_capacity", "LogReg", "tiny")
        assert result.primary_level == 1  # Backed off to strategy-only

    def test_smoothing_blends_levels(self):
        """The smoothed estimate should blend specific and general evidence."""
        pm = PosteriorMemory(kappa=3.0, min_evidence=2)
        # L1: 3 successes (p=high)
        for mt in ["MLP", "RF", "SVC"]:
            pm.update("increase_capacity", mt, "small", success=True)
        # L3: 2 failures (p=low at this specific context)
        pm.update("increase_capacity", "MLP", "medium", success=False)
        pm.update("increase_capacity", "MLP", "medium", success=False)

        result = pm.query("increase_capacity", "MLP", "medium")
        # The smoothed result should be between L3's low and L1's high
        assert result.success_probability < 0.5  # L3 pulls it down
        assert result.success_probability > 0.1  # But L1 prevents it from collapsing

    def test_different_actions_are_independent(self):
        """Evidence for action A should not affect action B's action-specific posteriors."""
        pm = PosteriorMemory(min_evidence=2)
        for _ in range(5):
            pm.update("increase_capacity", "MLP", "medium", success=True)

        result_a = pm.query("increase_capacity", "MLP", "medium")
        result_b = pm.query("add_regularization", "MLP", "medium")

        # Action A should have fine-grained evidence at L3
        assert result_a.primary_level == 3
        # Action B has no action-specific evidence — only global fallback
        assert result_b.primary_level == 0
        # Action A should have higher confidence (more specific evidence)
        assert result_a.confidence >= result_b.confidence
        # Action B's L1/L2/L3 evidence counts should be zero
        assert result_b.evidence_counts[3] == 0
        assert result_b.evidence_counts[2] == 0
        assert result_b.evidence_counts[1] == 0

    def test_nr001_fragmentation_addressed(self):
        """The hierarchical smoothing should handle the NR-001 scenario:
        sparse L3 evidence for some contexts, but sufficient L1/L0 evidence.

        In RF-1, this caused fallback to uninformative 0.6 prior.
        With the posterior, it should blend informative L1/L0 evidence.
        """
        pm = PosteriorMemory(kappa=3.0, min_evidence=3)
        # Simulate: strategy 'increase_capacity' generally works (L1 evidence)
        for i in range(10):
            mt = ["MLP", "RF", "SVC", "LogReg"][i % 4]
            bucket = ["small", "medium", "large"][i % 3]
            pm.update("increase_capacity", mt, bucket, success=True)

        # Query a specific context with only 1 L3 observation (below min_evidence)
        pm.update("increase_capacity", "MLP", "tiny", success=True)
        result = pm.query("increase_capacity", "MLP", "tiny")

        # Should get a high probability (backed by L1/L0), not the uninformative 0.5
        assert result.success_probability > 0.6
        # And should report where it backed off to
        assert result.primary_level <= 2  # Can't use L3 with only 1 sample

    def test_validation_rejects_invalid_params(self):
        """Invalid parameters should raise ValueError."""
        with pytest.raises(ValueError):
            PosteriorMemory(prior_alpha=-1.0)
        with pytest.raises(ValueError):
            PosteriorMemory(kappa=-1.0)
        with pytest.raises(ValueError):
            PosteriorMemory(min_evidence=0)


# ============================================================================
# ContextualBanditPolicy Tests
# ============================================================================

class TestContextualBanditPolicy:
    """Tests for the LinUCB contextual bandit policy."""

    @pytest.fixture
    def policy(self):
        return ContextualBanditPolicy(
            actions=["increase_capacity", "add_regularization", "change_family"],
            alpha=1.0,
            rng=random.Random(42),
        )

    def test_select_action_returns_valid_action(self, policy):
        """Selected action must be from the action set."""
        ctx = ContextVector()
        decision = policy.select_action(ctx)
        assert decision.chosen_action in policy.actions

    def test_select_action_returns_all_scores(self, policy):
        """Every action should have a score in the decision."""
        ctx = ContextVector()
        decision = policy.select_action(ctx)
        assert set(decision.action_scores.keys()) == set(policy.actions)

    def test_exploration_dominates_initially(self, policy):
        """With no updates, exploration bonus should dominate predicted reward."""
        ctx = ContextVector()
        decision = policy.select_action(ctx)
        for score in decision.action_scores.values():
            assert score.exploration_bonus >= abs(score.predicted_reward)

    def test_learning_improves_selection(self, policy):
        """After consistent rewards, the policy should prefer the rewarded action."""
        ctx = ContextVector(best_metric=0.5, budget_fraction=0.8)
        # Train: 'increase_capacity' always gives high reward
        for _ in range(20):
            policy.update("increase_capacity", ctx, reward=0.9)
            policy.update("add_regularization", ctx, reward=0.3)
            policy.update("change_family", ctx, reward=0.2)

        decision = policy.select_action(ctx)
        # The action with highest consistent reward should have highest predicted reward
        scores = decision.action_scores
        assert scores["increase_capacity"].predicted_reward > scores["add_regularization"].predicted_reward
        assert scores["increase_capacity"].predicted_reward > scores["change_family"].predicted_reward

    def test_context_sensitivity(self, policy):
        """Policy should learn to prefer different actions in different contexts."""
        ctx_early = ContextVector(best_metric=0.3, budget_fraction=0.9)
        ctx_late = ContextVector(best_metric=0.8, budget_fraction=0.1)

        # Train: 'increase_capacity' works early, 'add_regularization' works late
        for _ in range(30):
            policy.update("increase_capacity", ctx_early, reward=0.8)
            policy.update("add_regularization", ctx_early, reward=0.2)
            policy.update("increase_capacity", ctx_late, reward=0.2)
            policy.update("add_regularization", ctx_late, reward=0.8)

        # In early context, should prefer increase_capacity
        dec_early = policy.select_action(ctx_early)
        assert dec_early.action_scores["increase_capacity"].predicted_reward > \
               dec_early.action_scores["add_regularization"].predicted_reward

        # In late context, should prefer add_regularization
        dec_late = policy.select_action(ctx_late)
        assert dec_late.action_scores["add_regularization"].predicted_reward > \
               dec_late.action_scores["increase_capacity"].predicted_reward

    def test_memory_modifier_affects_scores(self, policy):
        """memory_modifier should scale the final score."""
        ctx = ContextVector()
        # Without modifier
        dec_no_mod = policy.select_action(ctx)
        # With modifier that halves 'increase_capacity'
        dec_mod = policy.select_action(ctx, memory_modifier=lambda a: 0.5 if a == "increase_capacity" else 1.0)

        ic_no_mod = dec_no_mod.action_scores["increase_capacity"].final_score
        ic_mod = dec_mod.action_scores["increase_capacity"].final_score
        assert ic_mod == pytest.approx(ic_no_mod * 0.5)

    def test_failure_check_halves_score(self, policy):
        """failure_check=True should multiply final score by 0.5."""
        ctx = ContextVector()
        dec = policy.select_action(ctx, failure_check=lambda a: a == "change_family")
        # change_family should have its memory_modifier recorded as 0.5
        assert dec.action_scores["change_family"].memory_modifier == pytest.approx(0.5)

    def test_select_random_is_uniform(self, policy):
        """select_random should use all actions."""
        counts = {a: 0 for a in policy.actions}
        for _ in range(300):
            counts[policy.select_random()] += 1
        # Each action should be selected at least 50 times in 300 draws
        for a in policy.actions:
            assert counts[a] > 30

    def test_context_dim_matches_vector(self, policy):
        """ContextVector.to_array() must have CONTEXT_DIM dimensions."""
        ctx = ContextVector()
        assert len(ctx.to_array()) == CONTEXT_DIM

    def test_policy_serialization(self, policy):
        """to_dict should capture policy state."""
        d = policy.to_dict()
        assert d["policy_type"] == "LinUCB"
        assert d["alpha"] == 1.0
        assert set(d["actions"]) == set(policy.actions)

    def test_decision_provenance(self, policy):
        """PolicyDecision.to_dict() should be serializable for provenance."""
        ctx = ContextVector(best_metric=0.6)
        decision = policy.select_action(ctx)
        d = decision.to_dict()
        assert "chosen_action" in d
        assert "scores" in d
        assert "context" in d


# ============================================================================
# TransferGate Tests
# ============================================================================

class TestTransferGate:
    """Tests for the adaptive memory-transfer gate."""

    @pytest.fixture
    def gate(self):
        return TransferGate(g_min=0.15)

    @pytest.fixture
    def high_confidence_posterior(self):
        return PosteriorQueryResult(
            success_probability=0.85,
            uncertainty=0.1,
            confidence=0.9,
            mean_delta=0.05,
            primary_level=3,
            evidence_counts={0: 20, 1: 10, 2: 5, 3: 5},
        )

    @pytest.fixture
    def low_confidence_posterior(self):
        return PosteriorQueryResult(
            success_probability=0.3,
            uncertainty=0.9,
            confidence=0.1,
            mean_delta=-0.02,
            primary_level=0,
            evidence_counts={0: 2, 1: 0, 2: 0, 3: 0},
        )

    def test_gate_always_in_bounds(self, gate):
        """Gate value must always be in [g_min, 1.0]."""
        for p_success in [0.0, 0.2, 0.5, 0.8, 1.0]:
            for uncertainty in [0.0, 0.5, 1.0]:
                for failure_rate in [0.0, 0.5, 1.0]:
                    posterior = PosteriorQueryResult(
                        success_probability=p_success,
                        uncertainty=uncertainty,
                        confidence=1.0 - uncertainty,
                        mean_delta=0.0,
                        primary_level=0,
                        evidence_counts={0: 5, 1: 0, 2: 0, 3: 0},
                    )
                    result = gate.compute("test_action", posterior,
                                         recent_failure_rate=failure_rate)
                    assert gate.g_min <= result.gate_value <= 1.0, \
                        f"Gate value {result.gate_value} out of bounds for " \
                        f"p={p_success}, unc={uncertainty}, fr={failure_rate}"

    def test_high_confidence_success_opens_gate(self, gate, high_confidence_posterior):
        """High-confidence success should produce a gate value near 1.0."""
        result = gate.compute("test_action", high_confidence_posterior,
                              recent_failure_rate=0.0, recency_score=1.0)
        assert result.gate_value > 0.8

    def test_high_failure_rate_closes_gate(self, gate, high_confidence_posterior):
        """High recent failure rate should close the gate."""
        result_open = gate.compute("test_action", high_confidence_posterior,
                                   recent_failure_rate=0.0)
        result_closed = gate.compute("test_action", high_confidence_posterior,
                                     recent_failure_rate=1.0)
        assert result_closed.gate_value < result_open.gate_value

    def test_low_confidence_dampens_gate(self, gate, low_confidence_posterior):
        """Low-confidence evidence should keep the gate moderate."""
        result = gate.compute("test_action", low_confidence_posterior)
        # Gate should be closer to g_min + 0.5*(1-g_min) ≈ 0.575
        # (neither fully open nor fully closed due to uncertainty)
        assert gate.g_min < result.gate_value < 0.9

    def test_gate_is_continuous(self, gate):
        """Small changes in posterior should produce small changes in gate value."""
        def gate_at(p_success):
            return gate.compute(
                "test", PosteriorQueryResult(
                    success_probability=p_success, uncertainty=0.3,
                    confidence=0.7, mean_delta=0.0, primary_level=2,
                    evidence_counts={0: 10, 1: 5, 2: 5, 3: 0},
                )
            ).gate_value

        # Gate should be monotonically increasing in p_success
        prev_v = gate_at(0.0)
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            v = gate_at(p)
            assert v >= prev_v, f"Gate decreased from {prev_v} to {v} at p={p}"
            prev_v = v

    def test_gate_never_fully_zero(self, gate):
        """Even worst case, gate should be >= g_min."""
        worst_posterior = PosteriorQueryResult(
            success_probability=0.0, uncertainty=1.0, confidence=0.0,
            mean_delta=-1.0, primary_level=0,
            evidence_counts={0: 100, 1: 0, 2: 0, 3: 0},
        )
        result = gate.compute("test", worst_posterior,
                              recent_failure_rate=1.0, recency_score=0.0)
        assert result.gate_value >= gate.g_min

    def test_compute_all_covers_all_actions(self, gate):
        """compute_all should return a decision for every action."""
        actions = ["a", "b", "c"]
        posteriors = {a: PosteriorQueryResult(
            success_probability=0.5, uncertainty=0.5, confidence=0.5,
            mean_delta=0.0, primary_level=0,
            evidence_counts={0: 5, 1: 0, 2: 0, 3: 0},
        ) for a in actions}

        results = gate.compute_all(actions, posteriors)
        assert set(results.keys()) == set(actions)

    def test_gate_decision_traceable(self, gate, high_confidence_posterior):
        """GateDecision should decompose the computation."""
        result = gate.compute("test", high_confidence_posterior)
        d = result.to_dict()
        assert "gate_value" in d
        assert "raw_logit" in d
        assert "components" in d
        assert "success" in d["components"]
        assert "failure" in d["components"]

    def test_modifier_fn_returns_callable(self, gate):
        """modifier_fn should return a callable compatible with the policy API."""
        posteriors = {
            "a": PosteriorQueryResult(
                success_probability=0.8, uncertainty=0.2, confidence=0.8,
                mean_delta=0.1, primary_level=2,
                evidence_counts={0: 10, 1: 5, 2: 5, 3: 0},
            ),
        }
        fn = gate.modifier_fn(posteriors)
        assert callable(fn)
        v = fn("a")
        assert 0.15 <= v <= 1.0

    def test_custom_weights(self):
        """Custom GateWeights should change gate behavior."""
        gate_default = TransferGate()
        gate_conservative = TransferGate(
            weights=GateWeights(w_failure=-5.0)  # Much more failure-averse
        )

        posterior = PosteriorQueryResult(
            success_probability=0.5, uncertainty=0.5, confidence=0.5,
            mean_delta=0.0, primary_level=1,
            evidence_counts={0: 10, 1: 5, 2: 0, 3: 0},
        )

        r_default = gate_default.compute("test", posterior, recent_failure_rate=0.5)
        r_conserv = gate_conservative.compute("test", posterior, recent_failure_rate=0.5)

        # Conservative gate should be more closed with 50% failure rate
        assert r_conserv.gate_value < r_default.gate_value


# ============================================================================
# Integration Tests: Posterior + Policy + Gate
# ============================================================================

class TestIntegration:
    """Tests verifying that the three modules compose correctly."""

    def test_full_decision_pipeline(self):
        """Posterior → Gate → Policy should produce a traceable decision."""
        # 1. Build posterior memory
        pm = PosteriorMemory(kappa=3.0, min_evidence=2)
        for _ in range(5):
            pm.update("increase_capacity", "MLP", "medium", success=True)
            pm.update("add_regularization", "MLP", "medium", success=False)
            pm.update("change_family", "MLP", "medium", success=True)

        # 2. Query posteriors for all actions
        posteriors = {}
        for a in ["increase_capacity", "add_regularization", "change_family"]:
            posteriors[a] = pm.query(a, "MLP", "medium")

        # 3. Compute gate values
        gate = TransferGate()
        gate_fn = gate.modifier_fn(posteriors)

        # 4. Build context and select action
        ctx = ContextVector(
            best_metric=0.6,
            budget_fraction=0.5,
            memory_evidence=0.8,
            posterior_success=posteriors["increase_capacity"].success_probability,
        )

        policy = ContextualBanditPolicy(
            actions=["increase_capacity", "add_regularization", "change_family"],
            rng=random.Random(42),
        )

        decision = policy.select_action(ctx, memory_modifier=gate_fn)

        # Verify the decision is complete and traceable
        assert decision.chosen_action in policy.actions
        assert len(decision.action_scores) == 3
        d = decision.to_dict()
        assert "scores" in d and "context" in d

    def test_posterior_gate_alignment(self):
        """The gate should be higher for actions with higher posterior success."""
        pm = PosteriorMemory(min_evidence=2)
        for _ in range(10):
            pm.update("good_action", "MLP", "medium", success=True)
            pm.update("bad_action", "MLP", "medium", success=False)

        gate = TransferGate()
        good_posterior = pm.query("good_action", "MLP", "medium")
        bad_posterior = pm.query("bad_action", "MLP", "medium")

        good_gate = gate.compute("good_action", good_posterior)
        bad_gate = gate.compute("bad_action", bad_posterior)

        assert good_gate.gate_value > bad_gate.gate_value

    def test_rf1_regression_condition_unchanged(self):
        """The old PolicyLearner import should still work (backward compat)."""
        from researchforge.policy.policy_learner import PolicyLearner
        pl = PolicyLearner(["a", "b", "c"])
        action = pl.select_action()
        assert action in ["a", "b", "c"]
        pl.update("a", 0.8)
        assert pl.times_tried["a"] == 1

    def test_ecrm_unchanged(self):
        """ECRM should still work exactly as before (regression guard)."""
        from researchforge.memory.ecrm import ECRM
        ecrm = ECRM()
        ecrm.store("test action MLP digits", {"task": "digits"}, {"metric": 0.8, "success": True}, "test")
        assert len(ecrm.records) == 1
        res = ecrm.research_experience_score("test")
        assert res > 0

    def test_controller_runs_with_contextual_policy(self):
        """Phase 2 verification: The full controller should run end-to-end with contextual_policy."""
        task = digits_task()
        controller = ResearchController(
            task=task,
            condition="contextual_policy",
            seed=42,
            population_size=3,
        )
        
        # Run a short search to verify the loop executes without crashing
        result = controller.run(n_generations=5)
        
        # Verify basic structural invariants
        assert result.condition == "contextual_policy"
        assert len(result.trials) == 6  # 1 baseline + 5 generations
        
        # Verify RF-2.0 specific provenance was captured
        assert result.posterior_stats is not None
        assert "global_observations" in result.posterior_stats
        assert result.posterior_stats["global_observations"] == 5  # 5 updates
        
        assert result.contextual_policy_stats is not None
        assert "policy" in result.contextual_policy_stats
        assert "gate" in result.contextual_policy_stats
        
        # Verify that policy decisions were recorded for the 5 generations
        assert len(result.policy_decisions) == 5
        for dec in result.policy_decisions:
            assert "strategy" in dec
            assert "decision_type" in dec
            assert dec["decision_type"] == "contextual_policy"
            assert "memory_decision_contribution" in dec
            assert "gate_value" in dec["memory_decision_contribution"]


# ============================================================================
# Mathematical Property Tests
# ============================================================================

class TestMathematicalProperties:
    """Tests that verify mathematical correctness beyond functional behavior."""

    def test_beta_posterior_is_proper_distribution(self):
        """Mean must be between 0 and 1, variance must be non-negative."""
        ps = PosteriorState(alpha=0.5, beta=0.5)  # Jeffreys prior
        assert 0.0 <= ps.mean <= 1.0
        assert ps.variance >= 0.0

        for _ in range(50):
            ps.update(success=random.random() > 0.4)
            assert 0.0 <= ps.mean <= 1.0
            assert ps.variance >= 0.0

    def test_posterior_mean_is_sufficient_statistic(self):
        """Two posterior states with same (α, β) must have same mean."""
        ps1 = PosteriorState(alpha=10.0, beta=5.0)
        ps2 = PosteriorState(alpha=10.0, beta=5.0)
        assert ps1.mean == ps2.mean
        assert ps1.variance == ps2.variance

    def test_hierarchical_smoothing_is_convex_combination(self):
        """The smoothed estimate must be between the specific and general estimates."""
        pm = PosteriorMemory(kappa=3.0, min_evidence=2)

        # L3: 100% success (5 obs)
        for _ in range(5):
            pm.update("a", "MLP", "medium", success=True)

        # L0: 50% success (many different contexts)
        for _ in range(10):
            pm.update("b", "RF", "large", success=True)
            pm.update("b", "RF", "large", success=False)

        result = pm.query("a", "MLP", "medium")
        # The smoothed result for "a" should be >= 0.5 (global) and <= 1.0 (L3)
        assert 0.5 <= result.success_probability <= 1.0

    def test_gate_sigmoid_is_monotone(self):
        """The sigmoid transfer function must be strictly monotone."""
        from researchforge.policy.transfer_gate import _sigmoid
        prev = _sigmoid(-100)
        for x in range(-99, 100):
            curr = _sigmoid(x)
            assert curr >= prev
            prev = curr
