"""researchforge/benchmarks/cohort/specification.py — Canonical Cohort Specification Factory.

Freezes and seals the BenchmarkAnalysisDatasetSpecification for Phase 12B:
- Tasks, task families, and transfer regime relations
- 6 experimental conditions with explicit capability declarations
- Fixed seeds [0, 1, 2, 3, 4] and orderings [forward, reverse, cross_first]
- Exact H1-H4 hypothesis contrasts and directions
- Derived execution budget (5,400 trials per condition, 32,400 total)
- Cryptographic SHA-256 sealing
"""
from __future__ import annotations

from typing import Dict

from .models import (
    BenchmarkCohortSpecification,
    BenchmarkCondition,
    CohortHypothesisContrast,
    ConditionCapabilitySpec,
    Direction,
    ExclusionPolicySpec,
    ExecutionBudget,
    SignConventionSpec,
    TaskFamily,
    TaskGenerationSpec,
    TransferRegime,
)


def create_canonical_cohort_spec() -> BenchmarkCohortSpecification:
    """Creates, configures, and cryptographically seals the Phase 12B Benchmark Cohort Specification."""
    # ── 1. Tasks & Deterministic Generation Specs ───────────────────────────────
    tasks: Dict[str, TaskGenerationSpec] = {}

    t_digits_0_4 = TaskGenerationSpec(
        task_id="digits_0_4",
        task_family=TaskFamily.DIGITS_SPATIAL,
        generation_algorithm="sklearn.datasets.load_digits:subset_0_4",
        generator_version="1.0.0",
        dataset_seed=42,
        sample_count_train=40,
        sample_count_val=160,
        sample_count_test=160,
        feature_dimensions=(64,),
        label_construction="raw_digit_classes_0_to_4",
        noise_parameters={"type": "none", "variance": 0.0},
        train_val_test_splits={"train_size": 40, "val_size": 160, "test_size": 160, "stratified": True},
        normalization="min_max_0_to_16_raw_pixel_integers",
        representation="flattened_8x8_grayscale_pixel_array",
        target_metric_name="accuracy",
        metric_direction=Direction.GREATER.value,
        target_metric_threshold=0.90,
    )
    t_digits_0_4 = TaskGenerationSpec(**{**t_digits_0_4.__dict__, "task_fingerprint": t_digits_0_4.compute_fingerprint()})
    tasks[t_digits_0_4.task_id] = t_digits_0_4

    t_digits_5_9 = TaskGenerationSpec(
        task_id="digits_5_9",
        task_family=TaskFamily.DIGITS_SPATIAL,
        generation_algorithm="sklearn.datasets.load_digits:subset_5_9",
        generator_version="1.0.0",
        dataset_seed=42,
        sample_count_train=40,
        sample_count_val=160,
        sample_count_test=160,
        feature_dimensions=(64,),
        label_construction="shifted_digit_classes_0_to_4_from_5_to_9",
        noise_parameters={"type": "none", "variance": 0.0},
        train_val_test_splits={"train_size": 40, "val_size": 160, "test_size": 160, "stratified": True},
        normalization="min_max_0_to_16_raw_pixel_integers",
        representation="flattened_8x8_grayscale_pixel_array",
        target_metric_name="accuracy",
        metric_direction=Direction.GREATER.value,
        target_metric_threshold=0.90,
    )
    t_digits_5_9 = TaskGenerationSpec(**{**t_digits_5_9.__dict__, "task_fingerprint": t_digits_5_9.compute_fingerprint()})
    tasks[t_digits_5_9.task_id] = t_digits_5_9

    t_digits_all_10 = TaskGenerationSpec(
        task_id="digits_all_10",
        task_family=TaskFamily.DIGITS_SPATIAL,
        generation_algorithm="sklearn.datasets.load_digits:full_10_class",
        generator_version="1.0.0",
        dataset_seed=42,
        sample_count_train=60,
        sample_count_val=200,
        sample_count_test=200,
        feature_dimensions=(64,),
        label_construction="raw_digit_classes_0_to_9",
        noise_parameters={"type": "none", "variance": 0.0},
        train_val_test_splits={"train_size": 60, "val_size": 200, "test_size": 200, "stratified": True},
        normalization="min_max_0_to_16_raw_pixel_integers",
        representation="flattened_8x8_grayscale_pixel_array",
        target_metric_name="accuracy",
        metric_direction=Direction.GREATER.value,
        target_metric_threshold=0.85,
    )
    t_digits_all_10 = TaskGenerationSpec(**{**t_digits_all_10.__dict__, "task_fingerprint": t_digits_all_10.compute_fingerprint()})
    tasks[t_digits_all_10.task_id] = t_digits_all_10

    t_ecg_lead1 = TaskGenerationSpec(
        task_id="synthetic_ecg_lead1",
        task_family=TaskFamily.ECG_TEMPORAL,
        generation_algorithm="researchforge.benchmarks.tasks:synthetic_ecg_task",
        generator_version="1.0.0",
        dataset_seed=42,
        sample_count_train=80,
        sample_count_val=310,
        sample_count_test=310,
        feature_dimensions=(96,),
        label_construction="binary_normal_vs_arrhythmia_0_1",
        noise_parameters={"type": "gaussian", "sigma": 0.30},
        train_val_test_splits={"total_samples": 700, "train_size": 80, "val_size": 310, "test_size": 310},
        normalization="continuous_zero_centered_waveform",
        representation="1d_temporal_signal_lead_1",
        target_metric_name="accuracy",
        metric_direction=Direction.GREATER.value,
        target_metric_threshold=0.82,
    )
    t_ecg_lead1 = TaskGenerationSpec(**{**t_ecg_lead1.__dict__, "task_fingerprint": t_ecg_lead1.compute_fingerprint()})
    tasks[t_ecg_lead1.task_id] = t_ecg_lead1

    t_ecg_lead2 = TaskGenerationSpec(
        task_id="synthetic_ecg_lead2",
        task_family=TaskFamily.ECG_TEMPORAL,
        generation_algorithm="researchforge.benchmarks.continuity.tasks:synthetic_ecg_lead2_task",
        generator_version="1.0.0",
        dataset_seed=42,
        sample_count_train=80,
        sample_count_val=310,
        sample_count_test=310,
        feature_dimensions=(96,),
        label_construction="binary_shifted_morphology_0_1",
        noise_parameters={"type": "gaussian_with_jitter", "sigma": 0.35, "jitter_range": 0.03},
        train_val_test_splits={"total_samples": 700, "train_size": 80, "val_size": 310, "test_size": 310},
        normalization="continuous_zero_centered_waveform",
        representation="1d_temporal_signal_lead_2",
        target_metric_name="accuracy",
        metric_direction=Direction.GREATER.value,
        target_metric_threshold=0.78,
    )
    t_ecg_lead2 = TaskGenerationSpec(**{**t_ecg_lead2.__dict__, "task_fingerprint": t_ecg_lead2.compute_fingerprint()})
    tasks[t_ecg_lead2.task_id] = t_ecg_lead2

    t_xor_parity = TaskGenerationSpec(
        task_id="xor_parity_8bit",
        task_family=TaskFamily.XOR_TABULAR,
        generation_algorithm="synthetic_noisy_parity_8bit_generator",
        generator_version="1.0.0",
        dataset_seed=42,
        sample_count_train=60,
        sample_count_val=270,
        sample_count_test=270,
        feature_dimensions=(8,),
        label_construction="odd_sum_parity_modulo_2",
        noise_parameters={"type": "additive_gaussian_perturbation", "sigma": 0.15},
        train_val_test_splits={"total_samples": 600, "train_size": 60, "val_size": 270, "test_size": 270},
        normalization="noisy_continuous_features",
        representation="tabular_discrete_continuous_vector",
        target_metric_name="accuracy",
        metric_direction=Direction.GREATER.value,
        target_metric_threshold=0.85,
    )
    t_xor_parity = TaskGenerationSpec(**{**t_xor_parity.__dict__, "task_fingerprint": t_xor_parity.compute_fingerprint()})
    tasks[t_xor_parity.task_id] = t_xor_parity

    task_families = {k: v.task_family for k, v in tasks.items()}

    # ── 2. Conditions & Capability Declarations ─────────────────────────────────
    conditions: Dict[str, ConditionCapabilitySpec] = {
        BenchmarkCondition.COLD_START.value: ConditionCapabilitySpec(
            condition_id=BenchmarkCondition.COLD_START,
            memory_enabled=True,
            trajectory_enabled=False,
            adaptive_retrieval_enabled=False,
            cross_task_experience_allowed=False,
            research_policy_learning_enabled=False,
            prior_task_information_available=False,
            description="Fresh researcher state initialized for every task with standard within-task machinery.",
        ),
        BenchmarkCondition.NO_MEMORY.value: ConditionCapabilitySpec(
            condition_id=BenchmarkCondition.NO_MEMORY,
            memory_enabled=False,
            trajectory_enabled=False,
            adaptive_retrieval_enabled=False,
            cross_task_experience_allowed=False,
            research_policy_learning_enabled=False,
            prior_task_information_available=False,
            description="Memory mechanisms explicitly disabled within and across tasks.",
        ),
        BenchmarkCondition.FLAT_ECRM.value: ConditionCapabilitySpec(
            condition_id=BenchmarkCondition.FLAT_ECRM,
            memory_enabled=True,
            trajectory_enabled=False,
            adaptive_retrieval_enabled=False,
            cross_task_experience_allowed=True,
            research_policy_learning_enabled=False,
            prior_task_information_available=True,
            description="Unfiltered flat empirical evidence (ECRM) accumulated across sequential prior tasks.",
        ),
        BenchmarkCondition.TRAJECTORY_MEMORY.value: ConditionCapabilitySpec(
            condition_id=BenchmarkCondition.TRAJECTORY_MEMORY,
            memory_enabled=True,
            trajectory_enabled=True,
            adaptive_retrieval_enabled=False,
            cross_task_experience_allowed=True,
            research_policy_learning_enabled=False,
            prior_task_information_available=True,
            description="Trajectory sequence memory accumulated and retrieved without confidence-based backoff.",
        ),
        BenchmarkCondition.ADAPTIVE_TRAJECTORY.value: ConditionCapabilitySpec(
            condition_id=BenchmarkCondition.ADAPTIVE_TRAJECTORY,
            memory_enabled=True,
            trajectory_enabled=True,
            adaptive_retrieval_enabled=True,
            cross_task_experience_allowed=True,
            research_policy_learning_enabled=False,
            prior_task_information_available=True,
            description="Context-sensitive adaptive trajectory retrieval with similarity thresholds and backoff.",
        ),
        BenchmarkCondition.CONTINUOUS_EXPERIENCE.value: ConditionCapabilitySpec(
            condition_id=BenchmarkCondition.CONTINUOUS_EXPERIENCE,
            memory_enabled=True,
            trajectory_enabled=True,
            adaptive_retrieval_enabled=True,
            cross_task_experience_allowed=True,
            research_policy_learning_enabled=True,
            prior_task_information_available=True,
            description="Unified continuous experience transfer with cross-task research-policy learning enabled.",
        ),
    }

    # ── 3. Exact Hypothesis Contrasts ──────────────────────────────────────────
    hypotheses: Dict[str, CohortHypothesisContrast] = {
        "H1": CohortHypothesisContrast(
            hypothesis_id="H1",
            title="Primary Confirmatory: Same-Family Experience Benefit",
            test_condition=BenchmarkCondition.CONTINUOUS_EXPERIENCE,
            control_condition=BenchmarkCondition.COLD_START,
            regime=TransferRegime.SAME_FAMILY,
            endpoint="decision_quality",
            direction=Direction.GREATER.value,
            contrast_formula="DQ(CONTINUOUS_EXPERIENCE, SAME_FAMILY) - DQ(COLD_START, NO_PRIOR_EXPERIENCE) > 0",
        ),
        "H2": CohortHypothesisContrast(
            hypothesis_id="H2",
            title="Secondary: Cross-Family Negative Transfer",
            test_condition=BenchmarkCondition.CONTINUOUS_EXPERIENCE,
            control_condition=BenchmarkCondition.COLD_START,
            regime=TransferRegime.CROSS_FAMILY,
            endpoint="decision_quality",
            direction=Direction.LESS.value,
            contrast_formula="DQ(CONTINUOUS_EXPERIENCE, CROSS_FAMILY) - DQ(COLD_START, NO_PRIOR_EXPERIENCE) < 0",
        ),
        "H3": CohortHypothesisContrast(
            hypothesis_id="H3",
            title="Secondary: Adaptive Retrieval Safety",
            test_condition=BenchmarkCondition.ADAPTIVE_TRAJECTORY,
            control_condition=BenchmarkCondition.TRAJECTORY_MEMORY,
            regime=TransferRegime.CROSS_FAMILY,
            endpoint="negative_transfer_rate",
            direction=Direction.LESS.value,
            contrast_formula="NTR(ADAPTIVE_TRAJECTORY, CROSS_FAMILY) - NTR(TRAJECTORY_MEMORY, CROSS_FAMILY) < 0",
        ),
        "H4": CohortHypothesisContrast(
            hypothesis_id="H4",
            title="Secondary: Non-Monotonic Experience Accumulation",
            test_condition=BenchmarkCondition.CONTINUOUS_EXPERIENCE,
            control_condition=BenchmarkCondition.COLD_START,
            regime=None,
            endpoint="decision_quality",
            direction=Direction.NON_MONOTONIC.value,
            contrast_formula="C1 = DQ_1 - DQ_0 > 0 and C2 = DQ_2 - DQ_1 < 0; Ccomp = DQ_1 - (DQ_0 + DQ_2) / 2 > 0",
            horizon_definition={
                "dq0": "Target evaluated with zero prior tasks (sequence position 0, fresh start)",
                "dq1": "Target evaluated after one eligible prior task (sequence position 1)",
                "dq2": "Target evaluated after two eligible prior tasks (sequence position 2)",
                "experience_condition": "CONTINUOUS_EXPERIENCE",
                "composite_contrast": "DQ_1 - (DQ_0 + DQ_2) / 2",
                "decision_rule": "H4 supported if C1 > 0 and C2 < 0 strictly under CONTINUOUS_EXPERIENCE",
            },
        ),
    }

    # ── 4. Execution Budget ─────────────────────────────────────────────────────
    budget = ExecutionBudget(
        n_tasks=len(tasks),
        n_seeds=5,
        n_orderings=3,
        n_conditions=len(conditions),
        generations_per_task=10,
        population_size=6,
        timeout_per_trial_seconds=30.0,
        trial_definition="one population-member evaluation (individual candidate genome evaluation)",
        wallclock_semantics="theoretical serialized upper bound assuming worst-case timeout per trial, not a predicted wall-clock runtime",
    )

    spec = BenchmarkCohortSpecification(
        specification_version="1.0.1",
        parent_fingerprint="b95ef8fa43a49f7d306e7e42a117b45c51186699467945597d06c2daf7c27707",
        superseded_fingerprints=("b95ef8fa43a49f7d306e7e42a117b45c51186699467945597d06c2daf7c27707",),
        title="Phase 12B Research Continuity Benchmark Analysis Dataset Specification",
        code_revision="HEAD",
        sign_convention=SignConventionSpec(),
        tasks=tasks,
        task_families=task_families,
        conditions=conditions,
        hypotheses=hypotheses,
        seeds=(0, 1, 2, 3, 4),
        orderings=("forward", "reverse", "cross_first"),
        budget=budget,
        exclusion_policy=ExclusionPolicySpec(),
    )
    return spec.seal()
