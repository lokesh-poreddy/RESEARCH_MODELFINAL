"""tests/test_phase12b_cohort_specification.py — Unit tests for Phase 12B Benchmark Cohort Specification.

Verifies:
1. Deep immutability (CohortMutationError on modification of sealed specification, dicts, or sequence)
2. Canonical sign convention (delta = test - control, GREATER, LESS, satisfaction check)
3. TaskFamily vs. TransferRegime separation (regime computed dynamically, not intrinsic to task)
4. TransferRegime determination relations (SAME_FAMILY, CROSS_FAMILY, UNRELATED)
5. Deterministic task generation contracts, universe completeness, and task fingerprints
6. Exact experimental conditions and explicit capability flags
7. Exact hypothesis contrasts (H1, H2, H3, H4 horizon construction)
8. Seeds and orderings immutability
9. Mathematically derived execution budget
10. Failure outcome taxonomy and pre-registered exclusion rules
11. Deterministic fingerprinting and tamper detection
12. Statistical Analysis Plan (SAP) and Cohort consistency
"""
import pytest

from researchforge.benchmarks.cohort import (
    BenchmarkCohortSpecification,
    BenchmarkCondition,
    CohortHypothesisContrast,
    CohortMutationError,
    ConditionCapabilitySpec,
    Direction,
    ExclusionPolicySpec,
    ExecutionBudget,
    FrozenDict,
    RunOutcomeCategory,
    SignConvention,
    SignConventionSpec,
    TaskFamily,
    TaskGenerationSpec,
    TransferRegime,
    create_canonical_cohort_spec,
    determine_sequence_transfer_regime,
    determine_transfer_regime,
)
from researchforge.benchmarks.statistical import create_canonical_sap


# ── 1. Immutability Tests ──────────────────────────────────────────────────────

def test_cohort_spec_deep_immutability():
    """Sealed cohort specification and nested structures must reject any modification."""
    spec = create_canonical_cohort_spec()

    assert spec.is_sealed is True
    assert spec.specification_version == "1.0.1"
    assert spec.parent_fingerprint == "b95ef8fa43a49f7d306e7e42a117b45c51186699467945597d06c2daf7c27707"
    assert "b95ef8fa43a49f7d306e7e42a117b45c51186699467945597d06c2daf7c27707" in spec.superseded_fingerprints
    assert spec.cohort_fingerprint != ""
    assert len(spec.cohort_fingerprint) == 64
    assert spec.cohort_fingerprint != spec.parent_fingerprint

    # Direct attribute mutation on sealed spec
    with pytest.raises(CohortMutationError) as exc:
        spec.title = "Altered Title"
    assert "sealed and immutable" in str(exc.value)

    # Nested dictionary mutation
    with pytest.raises(CohortMutationError):
        spec.tasks["digits_0_4"] = None

    with pytest.raises(CohortMutationError):
        del spec.conditions["COLD_START"]

    with pytest.raises(CohortMutationError):
        spec.hypotheses.clear()

    with pytest.raises(CohortMutationError):
        spec.task_families.update({"new_task": TaskFamily.DIGITS_SPATIAL})


# ── 2. Canonical Sign Convention Tests ────────────────────────────────────────

def test_canonical_sign_convention_contract():
    """Delta must be strictly test - control, evaluated against expected direction."""
    sign_spec = SignConventionSpec()
    assert sign_spec.convention_name == SignConvention.CANONICAL_TEST_MINUS_CONTROL
    assert sign_spec.formula == "delta = test_value - control_value"

    # Positive improvement on higher-is-better metric (e.g. DQ: test=0.90, control=0.80 -> delta=+0.10)
    delta_gain = SignConventionSpec.compute_delta(test_value=0.90, control_value=0.80)
    assert delta_gain == pytest.approx(0.10)
    assert SignConventionSpec.is_direction_satisfied(delta_gain, Direction.GREATER.value) is True
    assert SignConventionSpec.is_direction_satisfied(delta_gain, Direction.LESS.value) is False

    # Negative degradation on higher-is-better metric (e.g. Cross-family DQ: test=0.75, control=0.85 -> delta=-0.10)
    delta_loss = SignConventionSpec.compute_delta(test_value=0.75, control_value=0.85)
    assert delta_loss == pytest.approx(-0.10)
    assert SignConventionSpec.is_direction_satisfied(delta_loss, Direction.LESS.value) is True
    assert SignConventionSpec.is_direction_satisfied(delta_loss, Direction.GREATER.value) is False

    # Reduction on lower-is-better metric (e.g. Negative transfer rate: test=0.05, control=0.20 -> delta=-0.15)
    delta_ntr = SignConventionSpec.compute_delta(test_value=0.05, control_value=0.20)
    assert delta_ntr == pytest.approx(-0.15)
    assert SignConventionSpec.is_direction_satisfied(delta_ntr, Direction.LESS.value) is True


# ── 3. Task Family vs. Transfer Regime Separation ────────────────────────────

def test_task_family_and_transfer_regime_separation():
    """Tasks must only possess intrinsic TaskFamily; TransferRegime is purely relational."""
    spec = create_canonical_cohort_spec()

    # Verify task universe size and intrinsic families
    assert len(spec.tasks) == 6
    assert spec.task_families["digits_0_4"] == TaskFamily.DIGITS_SPATIAL
    assert spec.task_families["digits_5_9"] == TaskFamily.DIGITS_SPATIAL
    assert spec.task_families["digits_all_10"] == TaskFamily.DIGITS_SPATIAL
    assert spec.task_families["synthetic_ecg_lead1"] == TaskFamily.ECG_TEMPORAL
    assert spec.task_families["synthetic_ecg_lead2"] == TaskFamily.ECG_TEMPORAL
    assert spec.task_families["xor_parity_8bit"] == TaskFamily.XOR_TABULAR

    # Ensure no TaskGenerationSpec contains a hardcoded transfer regime field
    for task_spec in spec.tasks.values():
        assert not hasattr(task_spec, "regime")
        assert not hasattr(task_spec, "transfer_regime")
        assert hasattr(task_spec, "task_family")


def test_transfer_regime_deterministic_relations():
    """TransferRegime must be deterministically computed from source and target families."""
    # 1. Self / Same-Family / Baseline
    assert determine_transfer_regime(TaskFamily.DIGITS_SPATIAL, TaskFamily.DIGITS_SPATIAL) == TransferRegime.SAME_FAMILY
    assert determine_transfer_regime(TaskFamily.ECG_TEMPORAL, TaskFamily.ECG_TEMPORAL) == TransferRegime.SAME_FAMILY
    assert determine_transfer_regime(TaskFamily.XOR_TABULAR, TaskFamily.XOR_TABULAR) == TransferRegime.SAME_FAMILY
    # Zero prior experience must be strictly NO_PRIOR_EXPERIENCE, NOT SAME_FAMILY
    assert determine_transfer_regime(None, TaskFamily.DIGITS_SPATIAL) == TransferRegime.NO_PRIOR_EXPERIENCE
    assert determine_transfer_regime(None, TaskFamily.ECG_TEMPORAL) == TransferRegime.NO_PRIOR_EXPERIENCE
    assert determine_transfer_regime(None, TaskFamily.XOR_TABULAR) == TransferRegime.NO_PRIOR_EXPERIENCE

    # 2. Cross-Family (Spatial Digits <-> Temporal Waveform)
    assert determine_transfer_regime(TaskFamily.DIGITS_SPATIAL, TaskFamily.ECG_TEMPORAL) == TransferRegime.CROSS_FAMILY
    assert determine_transfer_regime(TaskFamily.ECG_TEMPORAL, TaskFamily.DIGITS_SPATIAL) == TransferRegime.CROSS_FAMILY

    # 3. Unrelated (Tabular XOR Parity vs Spatial or Temporal)
    assert determine_transfer_regime(TaskFamily.DIGITS_SPATIAL, TaskFamily.XOR_TABULAR) == TransferRegime.UNRELATED
    assert determine_transfer_regime(TaskFamily.XOR_TABULAR, TaskFamily.DIGITS_SPATIAL) == TransferRegime.UNRELATED
    assert determine_transfer_regime(TaskFamily.ECG_TEMPORAL, TaskFamily.XOR_TABULAR) == TransferRegime.UNRELATED
    assert determine_transfer_regime(TaskFamily.XOR_TABULAR, TaskFamily.ECG_TEMPORAL) == TransferRegime.UNRELATED


def test_sequence_transfer_regime_evaluation():
    """Sequence transfer regime must correctly categorize multi-task histories."""
    # Zero prior history -> NO_PRIOR_EXPERIENCE
    assert determine_sequence_transfer_regime([], TaskFamily.DIGITS_SPATIAL) == TransferRegime.NO_PRIOR_EXPERIENCE

    # Same family accumulated
    history_same = [TaskFamily.DIGITS_SPATIAL, TaskFamily.DIGITS_SPATIAL]
    assert determine_sequence_transfer_regime(history_same, TaskFamily.DIGITS_SPATIAL) == TransferRegime.SAME_FAMILY

    # Cross family accumulated
    history_cross = [TaskFamily.DIGITS_SPATIAL, TaskFamily.DIGITS_SPATIAL]
    assert determine_sequence_transfer_regime(history_cross, TaskFamily.ECG_TEMPORAL) == TransferRegime.CROSS_FAMILY

    # Unrelated accumulated
    history_unrelated = [TaskFamily.DIGITS_SPATIAL, TaskFamily.ECG_TEMPORAL]
    assert determine_sequence_transfer_regime(history_unrelated, TaskFamily.XOR_TABULAR) == TransferRegime.UNRELATED


# ── 4. Deterministic Task Generation Contracts ────────────────────────────────

def test_deterministic_task_generation_contracts():
    """Every task in the universe must have a fully deterministic contract and fingerprint."""
    spec = create_canonical_cohort_spec()

    expected_tasks = {
        "digits_0_4",
        "digits_5_9",
        "digits_all_10",
        "synthetic_ecg_lead1",
        "synthetic_ecg_lead2",
        "xor_parity_8bit",
    }
    assert set(spec.tasks.keys()) == expected_tasks

    for task_id, task in spec.tasks.items():
        assert task.task_id == task_id
        assert task.dataset_seed == 42
        assert task.sample_count_train > 0
        assert task.sample_count_val > 0
        assert task.sample_count_test > 0
        assert len(task.feature_dimensions) >= 1
        assert task.label_construction != ""
        assert task.normalization != ""
        assert task.representation != ""
        assert task.target_metric_name == "accuracy"
        assert task.metric_direction == "GREATER"
        assert task.task_fingerprint != ""
        assert len(task.task_fingerprint) == 64
        # Re-computing fingerprint matches stored fingerprint
        assert task.compute_fingerprint() == task.task_fingerprint


# ── 5. Experimental Conditions & Capability Flags ─────────────────────────────

def test_experimental_conditions_capability_flags():
    """All 6 conditions must have explicit, frozen capability specifications."""
    spec = create_canonical_cohort_spec()

    assert len(spec.conditions) == 6
    conds = spec.conditions

    # 1. COLD_START: within-task default memory enabled, no cross-task inheritance
    c_cold = conds["COLD_START"]
    assert c_cold.memory_enabled is True
    assert c_cold.trajectory_enabled is False
    assert c_cold.adaptive_retrieval_enabled is False
    assert c_cold.cross_task_experience_allowed is False
    assert c_cold.research_policy_learning_enabled is False
    assert c_cold.prior_task_information_available is False

    # 2. NO_MEMORY: memory completely disabled
    c_nomem = conds["NO_MEMORY"]
    assert c_nomem.memory_enabled is False
    assert c_nomem.trajectory_enabled is False
    assert c_nomem.adaptive_retrieval_enabled is False
    assert c_nomem.cross_task_experience_allowed is False
    assert c_nomem.research_policy_learning_enabled is False
    assert c_nomem.prior_task_information_available is False

    # 3. FLAT_ECRM: flat cross-task evidence, no trajectory
    c_flat = conds["FLAT_ECRM"]
    assert c_flat.memory_enabled is True
    assert c_flat.trajectory_enabled is False
    assert c_flat.cross_task_experience_allowed is True
    assert c_flat.prior_task_information_available is True

    # 4. TRAJECTORY_MEMORY: trajectory cross-task memory without adaptive backoff
    c_traj = conds["TRAJECTORY_MEMORY"]
    assert c_traj.memory_enabled is True
    assert c_traj.trajectory_enabled is True
    assert c_traj.adaptive_retrieval_enabled is False
    assert c_traj.cross_task_experience_allowed is True

    # 5. ADAPTIVE_TRAJECTORY: context-sensitive retrieval with backoff
    c_adapt = conds["ADAPTIVE_TRAJECTORY"]
    assert c_adapt.memory_enabled is True
    assert c_adapt.trajectory_enabled is True
    assert c_adapt.adaptive_retrieval_enabled is True
    assert c_adapt.cross_task_experience_allowed is True
    assert c_adapt.research_policy_learning_enabled is False

    # 6. CONTINUOUS_EXPERIENCE: full continuity with policy learning
    c_cont = conds["CONTINUOUS_EXPERIENCE"]
    assert c_cont.memory_enabled is True
    assert c_cont.trajectory_enabled is True
    assert c_cont.adaptive_retrieval_enabled is True
    assert c_cont.cross_task_experience_allowed is True
    assert c_cont.research_policy_learning_enabled is True
    assert c_cont.prior_task_information_available is True


# ── 6. Exact Hypothesis Contrasts ─────────────────────────────────────────────

def test_exact_hypothesis_contrasts():
    """Hypotheses H1-H4 must specify exact test, control, regime, endpoint, and direction."""
    spec = create_canonical_cohort_spec()

    # H1: Primary Confirmatory
    h1 = spec.hypotheses["H1"]
    assert h1.test_condition == BenchmarkCondition.CONTINUOUS_EXPERIENCE
    assert h1.control_condition == BenchmarkCondition.COLD_START
    assert h1.regime == TransferRegime.SAME_FAMILY
    assert h1.endpoint == "decision_quality"
    assert h1.direction == Direction.GREATER.value

    # H2: Secondary Cross-Family Negative Transfer
    h2 = spec.hypotheses["H2"]
    assert h2.test_condition == BenchmarkCondition.CONTINUOUS_EXPERIENCE
    assert h2.control_condition == BenchmarkCondition.COLD_START
    assert h2.regime == TransferRegime.CROSS_FAMILY
    assert h2.endpoint == "decision_quality"
    assert h2.direction == Direction.LESS.value

    # H3: Secondary Adaptive Retrieval Safety
    h3 = spec.hypotheses["H3"]
    assert h3.test_condition == BenchmarkCondition.ADAPTIVE_TRAJECTORY
    assert h3.control_condition == BenchmarkCondition.TRAJECTORY_MEMORY
    assert h3.regime == TransferRegime.CROSS_FAMILY
    assert h3.endpoint == "negative_transfer_rate"
    assert h3.direction == Direction.LESS.value

    # H4: Secondary Non-Monotonic Experience Accumulation
    h4 = spec.hypotheses["H4"]
    assert h4.test_condition == BenchmarkCondition.CONTINUOUS_EXPERIENCE
    assert h4.control_condition == BenchmarkCondition.COLD_START
    assert h4.direction == Direction.NON_MONOTONIC.value
    assert h4.horizon_definition is not None
    assert "dq0" in h4.horizon_definition
    assert "dq1" in h4.horizon_definition
    assert "dq2" in h4.horizon_definition
    assert "DQ_1 - (DQ_0 + DQ_2) / 2" in h4.horizon_definition["composite_contrast"]


# ── 7. Seeds and Orderings ────────────────────────────────────────────────────

def test_frozen_seeds_and_orderings():
    """Seeds and orderings must be frozen and immutable."""
    spec = create_canonical_cohort_spec()

    assert spec.seeds == (0, 1, 2, 3, 4)
    assert spec.orderings == ("forward", "reverse", "cross_first")
    assert isinstance(spec.seeds, tuple)
    assert isinstance(spec.orderings, tuple)


# ── 8. Derived Execution Budget ───────────────────────────────────────────────

def test_derived_execution_budget():
    """Execution budget must be mathematically derived from constituent parameters."""
    spec = create_canonical_cohort_spec()
    b = spec.budget

    assert b is not None
    assert b.n_tasks == 6
    assert b.n_seeds == 5
    assert b.n_orderings == 3
    assert b.n_conditions == 6
    assert b.generations_per_task == 10
    assert b.population_size == 6
    assert b.timeout_per_trial_seconds == 30.0
    assert "population-member" in b.trial_definition
    assert "theoretical serialized upper bound" in b.wallclock_semantics

    # Mathematical derivations
    assert b.sequences_per_condition == 5 * 3  # 15
    assert b.task_evaluations_per_condition == 6 * 15  # 90
    assert b.trials_per_task_evaluation == 10 * 6  # 60
    assert b.trials_per_condition == 90 * 60  # 5,400
    assert b.total_benchmark_trials == 6 * 5400  # 32,400
    assert b.max_wallclock_seconds_per_condition == 5400 * 30.0  # 162,000 s
    assert b.max_total_wallclock_seconds == 32400 * 30.0  # 972,000 s


# ── 9. Failure Semantics & Exclusion Policy ───────────────────────────────────

def test_failure_taxonomy_and_exclusion_policy():
    """Failure classes and pre-registered exclusion semantics must be explicit."""
    spec = create_canonical_cohort_spec()

    expected_categories = {
        "VALID_COMPLETED",
        "SCIENTIFIC_FAILURE",
        "EXECUTION_FAILURE",
        "INVALID_RUN",
        "INCONCLUSIVE_RUN",
    }
    assert set(spec.failure_categories) == expected_categories

    policy = spec.exclusion_policy
    assert policy.allow_post_hoc_task_removal is False
    assert policy.allow_post_hoc_seed_removal is False
    assert policy.allow_post_hoc_ordering_removal is False
    assert policy.allow_outlier_deletion is False
    assert policy.allow_bad_result_deletion is False
    assert policy.allow_manual_removal is False
    assert "Invalid execution artifacts are retained permanently" in policy.rule_statement


# ── 10. Deterministic Fingerprint & Tamper Detection ──────────────────────────

def test_cohort_fingerprint_determinism_and_tamper_detection():
    """Identical specifications produce identical hashes; modifications produce divergence."""
    spec1 = create_canonical_cohort_spec()
    spec2 = create_canonical_cohort_spec()

    assert spec1.cohort_fingerprint == spec2.cohort_fingerprint

    # Unsealed spec with altered seed set has different fingerprint
    unsealed = BenchmarkCohortSpecification(
        seeds=(0, 1),
        sign_convention=SignConventionSpec(),
        budget=spec1.budget,
        exclusion_policy=spec1.exclusion_policy,
    )
    assert unsealed.compute_fingerprint() != spec1.cohort_fingerprint


# ── 11. SAP and Cohort Specification Consistency ─────────────────────────────

def test_sap_and_cohort_consistency():
    """The Statistical Analysis Plan and Benchmark Cohort Specification must harmonize on hypotheses."""
    cohort = create_canonical_cohort_spec()
    sap = create_canonical_sap()

    assert cohort.sign_convention.convention_name.value == sap.sign_convention

    # Primary endpoint consistency
    h1_cohort = cohort.hypotheses["H1"]
    h1_sap = sap.hypotheses["H1"]
    assert h1_cohort.endpoint == sap.primary_endpoint
    assert h1_cohort.test_condition.value == h1_sap.test_condition
    assert h1_cohort.control_condition.value == h1_sap.control_condition
    assert h1_cohort.regime.value == h1_sap.regime_filter
    assert h1_cohort.direction == h1_sap.expected_direction

    # H2 consistency
    h2_cohort = cohort.hypotheses["H2"]
    h2_sap = sap.hypotheses["H2"]
    assert h2_cohort.direction == h2_sap.expected_direction
    assert h2_cohort.regime.value == h2_sap.regime_filter

    # H4 consistency
    h4_cohort = cohort.hypotheses["H4"]
    h4_sap = sap.hypotheses["H4"]
    assert h4_cohort.direction == h4_sap.expected_direction
