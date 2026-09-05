"""tests/test_phase11_continuity.py — Research Continuity Benchmark unit and integration tests.

Verifies:
1. COLD_START vs NO_MEMORY distinction
2. Sequential experience inheritance and memory reset
3. Controlled transfer classification using within-task baseline
4. Task ordering permutations (forward, reverse, cross_first)
5. Multi-seed raw observation preservation
6. Learning-curve telemetry across sequential task counts
7. Provenance tracing for transfer events
8. Deterministic replay from initial state, task definition, and seed
"""
import copy
import hashlib
import json
import pytest
from datetime import datetime, timezone

from researchforge.benchmarks.continuity import (
    ContinuityArtifact,
    ContinuityCondition,
    ContinuityEvaluator,
    ContinuityHarness,
    ContinuityObservation,
    ContinuityTask,
    FutureInformationLeakageError,
    TaskRegime,
    TransferClassification,
    TransferEvent,
    digits_0_4_task,
    digits_5_9_task,
    get_ordered_tasks,
    get_pilot_task_sequence,
    synthetic_ecg_lead1_task,
    tabular_xor_parity_task,
)
from researchforge.pipeline.controller import ResearchController


def test_cold_start_vs_no_memory_distinction():
    """Verify that COLD_START and NO_MEMORY are distinct conditions with different semantics."""
    task = digits_0_4_task(seed=42)

    # COLD_START: Memory is active, but starts with fresh empty state
    ctrl_cold = ResearchController(
        task.task,
        condition="cold_start",
        seed=42,
    )
    assert ctrl_cold.condition == "cold_start"
    assert ctrl_cold.use_memory is True
    assert ctrl_cold.memory_mode in ("ecrm", "trajectory", "adaptive")
    assert ctrl_cold.trajectory_memory is not None
    assert ctrl_cold.adaptive_trajectory_memory is not None
    initial_cold_fp = ctrl_cold.export_memory_fingerprint()

    # NO_MEMORY: Memory is explicitly disabled
    ctrl_nomem = ResearchController(
        task.task,
        condition="no_memory",
        seed=42,
    )
    assert ctrl_nomem.condition == "no_memory"
    assert ctrl_nomem.use_memory is False
    assert ctrl_nomem.memory_mode == "none"
    nomem_fp = ctrl_nomem.export_memory_fingerprint()
    assert nomem_fp == "no_memory_disabled"
    assert initial_cold_fp != nomem_fp


def test_experience_inheritance_and_reset():
    """Verify memory fingerprints change after execution, reset restores empty, and inheritance preserves state."""
    task = digits_0_4_task(seed=0)

    # Run task to accumulate memory
    ctrl1 = ResearchController(
        task.task,
        condition="continuous_experience",
        seed=0,
    )
    fp_before = ctrl1.export_memory_fingerprint()
    res1 = ctrl1.run(n_generations=2)
    fp_after = ctrl1.export_memory_fingerprint()
    assert fp_before != fp_after, "Memory fingerprint must change after accumulating experience"

    # Reset test
    ctrl1.reset_memory()
    fp_reset = ctrl1.export_memory_fingerprint()
    assert fp_reset == fp_before, "Resetting memory must restore initial empty state fingerprint"

    # Inheritance test: initialize a new controller inheriting memory from ctrl1 before reset
    ctrl1_again = ResearchController(
        task.task,
        condition="continuous_experience",
        seed=0,
    )
    ctrl1_again.run(n_generations=2)
    accumulated_fp = ctrl1_again.export_memory_fingerprint()

    # Task 2 inheriting from Task 1
    task2 = digits_5_9_task(seed=0)
    ctrl2 = ResearchController(
        task2.task,
        condition="continuous_experience",
        seed=0,
        ecrm=ctrl1_again.ecrm,
        trajectory_memory=ctrl1_again.trajectory_memory,
        adaptive_trajectory_memory=ctrl1_again.adaptive_trajectory_memory,
        policy_learner=ctrl1_again.policy_learner,
        failed_signatures=ctrl1_again._failed_signatures,
    )
    inherited_fp = ctrl2.export_memory_fingerprint()
    assert inherited_fp == accumulated_fp, "Inherited memory fingerprint must match source post-task fingerprint"


def test_controlled_transfer_classification():
    """Verify transfer is classified using controlled baseline and context, not raw delta alone."""
    evaluator = ContinuityEvaluator()

    # Case 1: Insufficient evidence (backed off to level 0 or no samples)
    evt_backed_off = evaluator.classify_transfer(
        source_task="digits_0_4",
        target_task="digits_5_9",
        strategy="random_forest_baseline",
        model_type="RandomForestClassifier",
        experienced_result=0.88,
        baseline_without_transfer=0.80,
        transfer_expected=True,
        provenance_id="prov_001",
        context_level=0,  # backed off to domain-general
    )
    assert evt_backed_off.transfer_classification == TransferClassification.INSUFFICIENT_EVIDENCE
    assert evt_backed_off.delta == 0.08

    # Case 2: Positive transfer (meaningful gain with context > 0)
    evt_positive = evaluator.classify_transfer(
        source_task="digits_0_4",
        target_task="digits_5_9",
        strategy="random_forest_baseline",
        model_type="RandomForestClassifier",
        experienced_result=0.85,
        baseline_without_transfer=0.80,
        transfer_expected=True,
        provenance_id="prov_002",
        context_level=2,
    )
    assert evt_positive.transfer_classification == TransferClassification.POSITIVE
    assert evt_positive.delta == 0.05

    # Case 3: Neutral transfer (within epsilon threshold)
    evt_neutral = evaluator.classify_transfer(
        source_task="digits_0_4",
        target_task="digits_5_9",
        strategy="random_forest_baseline",
        model_type="RandomForestClassifier",
        experienced_result=0.805,
        baseline_without_transfer=0.80,
        transfer_expected=True,
        provenance_id="prov_003",
        context_level=2,
    )
    assert evt_neutral.transfer_classification == TransferClassification.NEUTRAL

    # Case 4: Negative transfer (significant degradation relative to within-task baseline)
    evt_negative = evaluator.classify_transfer(
        source_task="digits_0_4",
        target_task="tabular_xor_parity",
        strategy="decision_tree_simple",
        model_type="DecisionTreeClassifier",
        experienced_result=0.68,
        baseline_without_transfer=0.75,
        transfer_expected=False,
        provenance_id="prov_004",
        context_level=2,
    )
    assert evt_negative.transfer_classification == TransferClassification.NEGATIVE
    assert evt_negative.delta == -0.07


def test_task_orderings():
    """Verify task ordering permutations forward, reverse, cross_first."""
    tasks = get_pilot_task_sequence(seed=0)
    assert len(tasks) == 3

    forward = get_ordered_tasks(tasks, "forward")
    assert [t.task_id for t in forward] == ["T_digits_0_4", "T_digits_5_9", "T_ecg_lead1"]

    reverse = get_ordered_tasks(tasks, "reverse")
    assert [t.task_id for t in reverse] == ["T_ecg_lead1", "T_digits_5_9", "T_digits_0_4"]

    cross_first = get_ordered_tasks(tasks, "cross_first")
    assert [t.task_id for t in cross_first] == ["T_ecg_lead1", "T_digits_0_4", "T_digits_5_9"]


def test_learning_curve_and_telemetry_aggregation():
    """Verify that prior_task_count telemetry increases sequentially and observations are stored."""
    harness = ContinuityHarness(
        tasks=get_pilot_task_sequence(seed=0),
        conditions=[ContinuityCondition.COLD_START, ContinuityCondition.CONTINUOUS_EXPERIENCE],
        orderings=["forward"],
        seeds=[0],
        n_generations=2,
    )
    artifact = harness.run()

    # Raw observations must preserve every evaluation point
    assert len(artifact.raw_observations) > 0
    # Must have observations for both conditions across 3 tasks = 6 observations
    assert len(artifact.raw_observations) == 6

    # Verify prior_task_count monotonically increases in continuous sequence
    cont_obs = [
        o for o in artifact.raw_observations
        if o["condition"] == "CONTINUOUS_EXPERIENCE" and o["order_id"] == "forward"
    ]
    assert len(cont_obs) == 3
    assert [o["prior_task_count"] for o in cont_obs] == [0, 1, 2]

    # Verify learning curve telemetry structure
    assert len(artifact.learning_curve_telemetry) > 0
    for lc in artifact.learning_curve_telemetry:
        assert "prior_task_count" in lc
        assert "mean_best_metric" in lc
        assert "mean_experience_gain" in lc
        assert "mean_decision_quality" in lc
        assert "sample_count" in lc


def test_within_task_transfer_control_diagnostic():
    """Verify within-task transfer control comparison: no prior vs relevant prior vs mismatched prior."""
    evaluator = ContinuityEvaluator()

    # Target task: digits_5_9
    # Control A: no prior experience (baseline = 0.82)
    # Control B: relevant prior experience (digits_0_4 -> 0.87, positive transfer)
    # Control C: mismatched prior experience (tabular_xor -> 0.76, negative transfer)
    evt_b = evaluator.classify_transfer(
        source_task="digits_0_4",
        target_task="digits_5_9",
        strategy="rf_opt",
        model_type="RandomForestClassifier",
        experienced_result=0.87,
        baseline_without_transfer=0.82,
        transfer_expected=True,
        provenance_id="prov_digits_transfer",
        context_level=2,
    )
    assert evt_b.transfer_classification == TransferClassification.POSITIVE

    evt_c = evaluator.classify_transfer(
        source_task="tabular_xor_parity",
        target_task="digits_5_9",
        strategy="xor_heuristic",
        model_type="DecisionTreeClassifier",
        experienced_result=0.76,
        baseline_without_transfer=0.82,
        transfer_expected=False,
        provenance_id="prov_mismatched_transfer",
        context_level=2,
    )
    assert evt_c.transfer_classification == TransferClassification.NEGATIVE


def test_provenance_integrity():
    """Verify provenance traceability of TransferEvent back to source and target tasks."""
    evt = TransferEvent(
        id="te_test_001",
        schema_version="1.0",
        source_task="digits_0_4",
        target_task="digits_5_9",
        strategy="gradient_boosting",
        model_type="GradientBoostingClassifier",
        baseline_without_transfer=0.78,
        experienced_result=0.84,
        delta=0.06,
        transfer_expected=True,
        transfer_classification=TransferClassification.POSITIVE,
        provenance_id="source_traj_42#decision_3",
        context_level=2,
        rationale="Shared 8x8 image spatial representation",
    )
    assert evt.source_task == "digits_0_4"
    assert evt.target_task == "digits_5_9"
    assert evt.provenance_id == "source_traj_42#decision_3"
    assert "8x8 image" in evt.rationale
    assert evt.transfer_classification == TransferClassification.POSITIVE


def test_deterministic_task_replay():
    """Verify a continuity task produces equivalent results when executed with identical seed and inputs."""
    task = digits_0_4_task(seed=77)

    ctrl_a = ResearchController(
        task.task,
        condition="cold_start",
        seed=77,
    )
    res_a = ctrl_a.run(n_generations=2)

    ctrl_b = ResearchController(
        task.task,
        condition="cold_start",
        seed=77,
    )
    res_b = ctrl_b.run(n_generations=2)

    assert res_a.best_metric == pytest.approx(res_b.best_metric, abs=1e-5)
    assert ctrl_a.export_memory_fingerprint() == ctrl_b.export_memory_fingerprint()
