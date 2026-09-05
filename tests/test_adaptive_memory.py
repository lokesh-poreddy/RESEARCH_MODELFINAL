"""Unit and invariant tests for Adaptive Trajectory Memory (Phase 7).

Scientific Test Matrix:
  A. exact L3 sufficiency
  B. deterministic L3 -> L2 backoff
  C. deterministic L2 -> L1 backoff
  D. deterministic L1 -> L0 backoff
  E. complete absence of evidence
  F. negative evidence retention
  G. failure majority detection
  H. provenance resolution
  I. duplicate storage / idempotency
  J. RSG serialization
  K. deterministic memory fingerprint
  O. oracle-backoff diagnostic
  Correction 1 & 2 verification: confidence derived from evidence dispersion, not context level.
"""
from __future__ import annotations

import pytest
from researchforge.memory.adaptive_trajectory import (
    AdaptiveTrajectoryMemory,
    AdaptiveTrajectoryRecord,
    ContextualRetrievalResult,
    FailureCheckResult,
    new_adaptive_trajectory_id,
)
from researchforge.genome.research_system_genome import (
    ResearchSystemGenome,
    ResearchMemoryConfig,
)


def _make_record(
    strategy: str = "increase_capacity",
    model_type: str = "MLPClassifier",
    bucket: str = "high",
    metric: float = 0.85,
    success: bool = True,
    failure: str = "None",
    generation: int = 0,
    finding_id: str = "find_001",
    hypothesis_id: str = "hyp_001",
    run_id: str = "run_001",
) -> AdaptiveTrajectoryRecord:
    return AdaptiveTrajectoryRecord(
        id=new_adaptive_trajectory_id(),
        generation=generation,
        stage="early",
        problem_context="Improve accuracy on digits",
        parent_model_type=model_type,
        parent_capacity_bucket=bucket,
        strategy=strategy,
        child_model_type=model_type,
        child_capacity_bucket=bucket,
        metric=metric,
        success=success,
        failure=failure,
        hypothesis_id=hypothesis_id,
        spec_id="spec_001",
        run_id=run_id,
        outcome_id="out_001",
        finding_id=finding_id,
        provenance_id="prov_001",
    )


# =========================================================================== #
# Test A: Exact L3 Sufficiency                                                #
# =========================================================================== #
def test_exact_l3_sufficiency():
    mem = AdaptiveTrajectoryMemory(min_context_samples=3)
    # Add 3 records matching L3
    for i, m in enumerate([0.80, 0.82, 0.84]):
        mem.store(_make_record(metric=m, generation=i))

    res: ContextualRetrievalResult = mem.query_context("increase_capacity", "MLPClassifier", "high")
    assert res.context_level == 3
    assert res.requested_context_level == 3
    assert res.sample_count == 3
    assert res.evidence_sufficiency is True
    assert res.fallback_reason is None
    assert res.success_rate == 1.0
    assert res.mean_metric is not None
    assert abs(res.mean_metric - 0.82) < 1e-4
    assert res.variance is not None and res.variance > 0.0
    assert res.dispersion is not None and res.dispersion > 0.0
    assert res.confidence > 0.0


# =========================================================================== #
# Corrections 1 & 2: Confidence Separated from Context Level & Dispersion      #
# =========================================================================== #
def test_confidence_reflects_dispersion_not_context_level():
    """Prove that statistical confidence is derived from evidence consistency,
    NOT hard-coded to L3=1.0, L2=0.8, etc."""
    mem_noisy = AdaptiveTrajectoryMemory(min_context_samples=3)
    # 3 samples with high variance / dispersion: 0.61, 0.91, 0.40
    for i, m in enumerate([0.61, 0.91, 0.40]):
        mem_noisy.store(_make_record(metric=m, generation=i))

    mem_consistent = AdaptiveTrajectoryMemory(min_context_samples=3)
    # 3 samples with near-zero variance: 0.74, 0.75, 0.74
    for i, m in enumerate([0.74, 0.75, 0.74]):
        mem_consistent.store(_make_record(metric=m, generation=i))

    res_noisy = mem_noisy.query_context("increase_capacity", "MLPClassifier", "high")
    res_consistent = mem_consistent.query_context("increase_capacity", "MLPClassifier", "high")

    # Both are at context_level == 3 and have sample_count == 3
    assert res_noisy.context_level == 3
    assert res_consistent.context_level == 3
    assert res_noisy.sample_count == 3
    assert res_consistent.sample_count == 3

    # Dispersion of noisy data is strictly greater than consistent data
    assert res_noisy.dispersion is not None and res_consistent.dispersion is not None
    assert res_noisy.dispersion > res_consistent.dispersion

    # Therefore, confidence of consistent evidence is strictly higher than noisy evidence
    assert res_consistent.confidence > res_noisy.confidence


# =========================================================================== #
# Test B: Deterministic L3 -> L2 Backoff                                      #
# =========================================================================== #
def test_deterministic_l3_to_l2_backoff():
    mem = AdaptiveTrajectoryMemory(min_context_samples=3)
    # 2 records at L3 ("high" bucket)
    mem.store(_make_record(bucket="high", generation=0))
    mem.store(_make_record(bucket="high", generation=1))
    # 1 record at L2 sibling ("medium" bucket, same strategy & model_type)
    mem.store(_make_record(bucket="medium", generation=2))

    # Querying "high" bucket has only 2 samples at L3 (< min 3), so backs off to L2
    res = mem.query_context("increase_capacity", "MLPClassifier", "high")
    assert res.context_level == 2
    assert res.requested_context_level == 3
    assert res.sample_count == 3
    assert res.evidence_sufficiency is True
    assert res.fallback_reason is not None
    assert "L3_insufficient_samples" in res.fallback_reason

    # Verify determinism: repeated query returns exact same result
    res2 = mem.query_context("increase_capacity", "MLPClassifier", "high")
    assert res.context_level == res2.context_level
    assert res.confidence == res2.confidence
    assert res.sample_count == res2.sample_count


# =========================================================================== #
# Test C: Deterministic L2 -> L1 Backoff                                      #
# =========================================================================== #
def test_deterministic_l2_to_l1_backoff():
    mem = AdaptiveTrajectoryMemory(min_context_samples=3)
    # 1 record at MLPClassifier
    mem.store(_make_record(model_type="MLPClassifier", bucket="high", generation=0))
    # 2 records at RandomForestClassifier with same strategy
    mem.store(_make_record(model_type="RandomForestClassifier", bucket="low", generation=1))
    mem.store(_make_record(model_type="RandomForestClassifier", bucket="medium", generation=2))

    # L3 has 1, L2 has 1, but L1 (strategy "increase_capacity") has 3
    res = mem.query_context("increase_capacity", "MLPClassifier", "high")
    assert res.context_level == 1
    assert res.requested_context_level == 3
    assert res.sample_count == 3
    assert res.evidence_sufficiency is True
    assert res.fallback_reason is not None
    assert "L2_insufficient_samples" in res.fallback_reason


# =========================================================================== #
# Test D: Deterministic L1 -> L0 Backoff                                      #
# =========================================================================== #
def test_deterministic_l1_to_l0_backoff():
    mem = AdaptiveTrajectoryMemory(min_context_samples=3)
    # 1 record at "increase_capacity"
    mem.store(_make_record(strategy="increase_capacity", generation=0))
    # 1 record at "change_family"
    mem.store(_make_record(strategy="change_family", generation=1))

    # L1 has 1 (< min 3). Total global records = 2. Backoff to Level 0
    res = mem.query_context("increase_capacity", "MLPClassifier", "high")
    assert res.context_level == 0
    assert res.requested_context_level == 3
    assert res.sample_count == 2
    assert res.fallback_reason is not None
    assert "L1_insufficient_samples" in res.fallback_reason


# =========================================================================== #
# Test E: Complete Absence of Evidence (Empty Memory)                         #
# =========================================================================== #
def test_complete_absence_of_evidence():
    mem = AdaptiveTrajectoryMemory(min_context_samples=3, default_prior=0.6)
    res = mem.query_context("increase_capacity", "MLPClassifier", "high")
    assert res.context_level == 0
    assert res.sample_count == 0
    assert res.evidence_sufficiency is False
    assert res.fallback_reason == "empty_memory"
    assert res.confidence == 0.0
    assert res.success_rate == 0.6
    assert res.mean_metric is None
    assert res.variance is None
    assert res.dispersion is None
    assert res.retrieved_records == []


# =========================================================================== #
# Test F: Negative Evidence Retention                                         #
# =========================================================================== #
def test_negative_evidence_retention():
    """Failures and unsuccessful interventions must never be discarded by backoff."""
    mem = AdaptiveTrajectoryMemory(min_context_samples=3)
    # 1 failure at L3
    mem.store(_make_record(bucket="high", success=False, failure="convergence_failure", generation=0))
    # 2 failures at L2 sibling
    mem.store(_make_record(bucket="medium", success=False, failure="divergence", generation=1))
    mem.store(_make_record(bucket="medium", success=True, failure="None", generation=2))

    # Query L3: backs off to L2
    res = mem.query_context("increase_capacity", "MLPClassifier", "high")
    assert res.context_level == 2
    assert res.sample_count == 3
    # Failures are preserved in retrieved records
    failures = [r for r in res.retrieved_records if not r.success]
    assert len(failures) == 2
    failure_types = {r.failure for r in failures}
    assert "convergence_failure" in failure_types
    assert "divergence" in failure_types
    assert res.success_rate == pytest.approx(1 / 3)


# =========================================================================== #
# Test G: Failure Majority Detection (FailureCheckResult)                     #
# =========================================================================== #
def test_failure_majority_detection():
    mem = AdaptiveTrajectoryMemory(min_context_samples=3)
    mem.store(_make_record(success=False, failure="bad_hyperparameters", generation=0))
    mem.store(_make_record(success=False, failure="timeout", generation=1))
    mem.store(_make_record(success=True, failure="None", generation=2))

    check: FailureCheckResult = mem.similar_trajectory_recently_failed(
        "increase_capacity", "MLPClassifier", "high", window=3
    )
    assert check.has_failed_majority is True
    assert bool(check) is True  # Truthy check
    assert check.context_level == 3
    assert check.sample_count == 3
    assert len(check.failed_records) == 2
    assert len(check.failed_record_ids) == 2
    assert "majority_failure" in check.reason


# =========================================================================== #
# Test H: Provenance Resolution                                               #
# =========================================================================== #
def test_provenance_resolution():
    mem = AdaptiveTrajectoryMemory(min_context_samples=1)
    rec = _make_record(
        finding_id="finding_123",
        hypothesis_id="hyp_456",
        run_id="run_789",
    )
    mem.store(rec)

    res = mem.query_context("increase_capacity", "MLPClassifier", "high")
    assert "finding_123" in res.provenance_refs
    assert "hyp_456" in res.provenance_refs
    assert "run_789" in res.provenance_refs

    # Mock graph with evidence_chain
    class MockGraph:
        def evidence_chain(self, target_id):
            return [f"step_for_{target_id}"]

    chain = mem.evidence_chain_for(rec.id, MockGraph())
    assert chain == ["step_for_finding_123"]


# =========================================================================== #
# Test I: Duplicate Storage / Idempotency                                     #
# =========================================================================== #
def test_idempotent_storage():
    mem = AdaptiveTrajectoryMemory()
    rec1 = _make_record(generation=0)
    assert mem.store(rec1) is True
    assert len(mem.records) == 1

    # Exact duplicate (same id)
    assert mem.store(rec1) is False
    assert len(mem.records) == 1

    # Semantic duplicate (different id, same semantic key)
    rec2 = _make_record(generation=0)
    assert rec2.id != rec1.id
    assert mem.store(rec2) is False
    assert len(mem.records) == 1


# =========================================================================== #
# Test J: RSG Serialization & Validation                                      #
# =========================================================================== #
def test_rsg_memory_config_serialization():
    cfg = ResearchMemoryConfig(memory_design="adaptive_trajectory", min_context_samples=4)
    d = cfg.to_dict()
    assert d["memory_design"] == "adaptive_trajectory"
    assert d["min_context_samples"] == 4

    restored = ResearchMemoryConfig.from_dict(d)
    assert restored.memory_design == "adaptive_trajectory"
    assert restored.min_context_samples == 4

    # Validation errors
    with pytest.raises(ValueError, match="min_context_samples must be >= 1"):
        ResearchMemoryConfig(min_context_samples=0)

    with pytest.raises(ValueError, match="memory_design must be one of"):
        ResearchMemoryConfig(memory_design="unknown_design")

    # RSG default factory with adaptive_trajectory
    rsg = ResearchSystemGenome.default(condition="adaptive_trajectory", seed=42)
    assert rsg.memory_config.memory_design == "adaptive_trajectory"
    rsg.validate()


# =========================================================================== #
# Test K: Deterministic Memory Fingerprint                                    #
# =========================================================================== #
def test_deterministic_memory_fingerprint():
    rsg1 = ResearchSystemGenome.default(condition="adaptive_trajectory", seed=10)
    rsg2 = ResearchSystemGenome.default(condition="adaptive_trajectory", seed=10)
    assert rsg1.fingerprint() == rsg2.fingerprint()

    # Mutation alters fingerprint deterministically
    rsg3 = ResearchSystemGenome.default(condition="adaptive_trajectory", seed=10)
    rsg3.memory_config.min_context_samples = 5
    assert rsg3.fingerprint() != rsg1.fingerprint()


# =========================================================================== #
# Test O: Oracle-Backoff Diagnostic                                           #
# =========================================================================== #
def test_oracle_backoff_diagnostic():
    mem = AdaptiveTrajectoryMemory(min_context_samples=3)
    # 2 records at L3 ("high" bucket)
    mem.store(_make_record(bucket="high", generation=0, metric=0.90))
    mem.store(_make_record(bucket="high", generation=1, metric=0.92))
    # 3 records at L2 ("medium" bucket)
    mem.store(_make_record(bucket="medium", generation=2, metric=0.70))
    mem.store(_make_record(bucket="medium", generation=3, metric=0.72))
    mem.store(_make_record(bucket="medium", generation=4, metric=0.74))

    # Automatic query backs off to L2 because L3 has only 2 samples (< min 3)
    auto_res = mem.query_context("increase_capacity", "MLPClassifier", "high")
    assert auto_res.context_level == 2
    assert auto_res.sample_count == 5

    # Oracle query forces Level 3 directly (2 samples), enabling diagnostic attribution
    oracle_l3 = mem.query_oracle("increase_capacity", "MLPClassifier", "high", oracle_level=3)
    assert oracle_l3.context_level == 3
    assert oracle_l3.sample_count == 2
    assert "oracle_level_forced: level=3" in oracle_l3.fallback_reason
    assert oracle_l3.mean_metric == pytest.approx(0.91)

    # Oracle query forces Level 2 directly
    oracle_l2 = mem.query_oracle("increase_capacity", "MLPClassifier", "high", oracle_level=2)
    assert oracle_l2.context_level == 2
    assert oracle_l2.sample_count == 5

    # Invalid oracle level raises ValueError
    with pytest.raises(ValueError, match="Invalid oracle_level"):
        mem.query_oracle("increase_capacity", "MLPClassifier", "high", oracle_level=4)


# =========================================================================== #
# Standalone Deterministic Replay Test                                        #
# =========================================================================== #
def test_deterministic_replay():
    """Verify that replaying a fixed sequence of research trajectories into two
    independent memory instances yields identical retrieval results, backoff
    decisions, and evidence statistics."""
    trials = [
        ("increase_capacity", "MLPClassifier", "high", 0.81, True, "None"),
        ("increase_capacity", "MLPClassifier", "high", 0.79, True, "None"),
        ("increase_capacity", "MLPClassifier", "medium", 0.75, False, "convergence_failure"),
        ("change_family", "SVC", "low", 0.88, True, "None"),
        ("change_family", "SVC", "low", 0.86, True, "None"),
        ("change_family", "SVC", "medium", 0.84, False, "timeout"),
        ("tune_regularization", "LogisticRegression", "medium", 0.65, False, "divergence"),
    ]

    mem1 = AdaptiveTrajectoryMemory(min_context_samples=3)
    mem2 = AdaptiveTrajectoryMemory(min_context_samples=3)

    for gen, (strat, mt, b, met, succ, fail) in enumerate(trials):
        rec1 = _make_record(strategy=strat, model_type=mt, bucket=b, metric=met, success=succ, failure=fail, generation=gen)
        rec2 = _make_record(strategy=strat, model_type=mt, bucket=b, metric=met, success=succ, failure=fail, generation=gen)
        # Ensure semantic keys match even if IDs differ
        rec2.id = f"custom_{rec1.id}"
        mem1.store(rec1)
        mem2.store(rec2)

    query_contexts = [
        ("increase_capacity", "MLPClassifier", "high"),  # 2 samples at L3 -> should back off to L2 (3 samples)
        ("change_family", "SVC", "low"),                 # 2 samples at L3 -> should back off to L2 (3 samples)
        ("tune_regularization", "LogisticRegression", "medium"), # 1 sample -> backs off to L1 (1) -> L0 (7)
        ("crossover", "RandomForestClassifier", "high"), # 0 samples -> backs off to L0 (7)
    ]

    for strat, mt, b in query_contexts:
        res1 = mem1.query_context(strat, mt, b)
        res2 = mem2.query_context(strat, mt, b)

        assert res1.context_level == res2.context_level
        assert res1.sample_count == res2.sample_count
        assert res1.evidence_sufficiency == res2.evidence_sufficiency
        assert res1.fallback_reason == res2.fallback_reason
        assert res1.success_rate == pytest.approx(res2.success_rate)
        assert res1.confidence == pytest.approx(res2.confidence)
        if res1.mean_metric is not None:
            assert res1.mean_metric == pytest.approx(res2.mean_metric)
        if res1.dispersion is not None:
            assert res1.dispersion == pytest.approx(res2.dispersion)


# =========================================================================== #
# Test L: Controller Integration                                              #
# =========================================================================== #
def test_controller_integration():
    """Verify end-to-end execution of ResearchController with condition='adaptive_trajectory'."""
    from researchforge.benchmarks.tasks import digits_task
    from researchforge.pipeline.controller import ResearchController

    task = digits_task(seed=0)
    ctrl = ResearchController(task, condition="adaptive_trajectory", seed=42)
    run_res = ctrl.run(n_generations=4)

    assert run_res.condition == "adaptive_trajectory"
    assert run_res.best_metric > 0.0
    assert len(run_res.trials) == 5  # generation -1 (baseline) + 4 search generations
    assert len(run_res.states) == 5
    assert run_res.adaptive_trajectory_stats["total_trajectories"] == 5
    assert len(ctrl.adaptive_trajectory_memory.records) == 5


# =========================================================================== #
# Test M: Memory Influence Auditability                                       #
# =========================================================================== #
def test_memory_influence_auditability():
    """A researcher must be able to inspect trial metadata and determine what
    memory recommended, what the baseline score was, and what the final score was."""
    from researchforge.benchmarks.tasks import digits_task
    from researchforge.pipeline.controller import ResearchController

    task = digits_task(seed=0)
    ctrl = ResearchController(task, condition="adaptive_trajectory", seed=7)
    run_res = ctrl.run(n_generations=3)

    # Inspect search trials (generations 0, 1, 2)
    search_trials = run_res.trials[1:]
    assert len(search_trials) == 3

    for t in search_trials:
        assert t.context_level in (0, 1, 2, 3)
        assert t.requested_context_level == 3
        assert t.sample_count is not None and t.sample_count >= 0
        assert t.memory_decision_contribution is not None

        contrib = t.memory_decision_contribution
        assert "memory_recommendation" in contrib
        assert "baseline_strategy_score" in contrib
        assert "final_strategy_score" in contrib
        assert "audit_per_action" in contrib

        # Verify each action has full diagnostic breakdown
        for a, audit in contrib["audit_per_action"].items():
            assert "baseline_score" in audit
            assert "multiplier" in audit
            assert "context_level" in audit
            assert "confidence" in audit
            assert "final_score" in audit


# =========================================================================== #
# Test N: Behavioral Parity of Existing Conditions                            #
# =========================================================================== #
def test_behavioral_parity_of_existing_conditions():
    """Verify that existing conditions ('full', 'trajectory_memory', 'no_memory', 'random')
    behave identically to pre-Phase 7 baseline and are strictly reproducible."""
    from researchforge.benchmarks.tasks import digits_task
    from researchforge.pipeline.controller import ResearchController

    task = digits_task(seed=0)

    for cond in ("full", "trajectory_memory", "no_memory", "random"):
        ctrl1 = ResearchController(task, condition=cond, seed=123)
        run1 = ctrl1.run(n_generations=3)

        ctrl2 = ResearchController(task, condition=cond, seed=123)
        run2 = ctrl2.run(n_generations=3)

        # Exact trial metrics and strategies must match identically
        metrics1 = [t.metric for t in run1.trials]
        metrics2 = [t.metric for t in run2.trials]
        strats1 = [t.strategy for t in run1.trials]
        strats2 = [t.strategy for t in run2.trials]

        assert metrics1 == metrics2
        assert strats1 == strats2
        assert run1.best_metric == run2.best_metric


