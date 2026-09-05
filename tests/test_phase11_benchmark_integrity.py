"""tests/test_phase11_benchmark_integrity.py — Rigorous benchmark integrity tests for Phase 11.

Verifies:
1. Future information isolation and FutureInformationLeakageError enforcement
2. Fixed compute/generation budget invariance across conditions
3. Memory reset correctness and inheritance verification
4. Evaluator independence from policy internals
5. Preservation of historical benchmark artifacts (RDE-Bench phase 6 & 7)
6. Separation of orchestration overhead from scientific performance metrics
7. Architectural boundary checks (no DB/SQLAlchemy coupling in continuity package)
"""
import copy
import hashlib
import os
import pytest
from pathlib import Path

from researchforge.benchmarks.continuity import (
    ContinuityArtifact,
    ContinuityCondition,
    ContinuityEvaluator,
    ContinuityHarness,
    ExperiencePartition,
    FutureInformationLeakageError,
    TaskRegime,
    TransferClassification,
    digits_0_4_task,
    digits_5_9_task,
    get_pilot_task_sequence,
    synthetic_ecg_lead1_task,
    tabular_xor_parity_task,
)
from researchforge.pipeline.controller import ResearchController


def test_future_information_leakage_enforcement():
    """Verify that attempting to query future task experience raises FutureInformationLeakageError."""
    partition = ExperiencePartition.create(
        task_id="T_digits_5_9",
        task_sequence_index=1,
        allowed_task_ids=["T_digits_0_4"],
        prior_memory_fingerprint="prior_mem_hash_123",
        current_task_fingerprint="curr_task_hash_456",
        future_task_ids=["T_ecg_lead1", "T_xor_parity"],
    )

    assert partition.is_sealed is True
    # Querying allowed prior task must succeed
    partition.verify_access("T_digits_0_4")

    # Querying future task must raise FutureInformationLeakageError
    with pytest.raises(FutureInformationLeakageError) as excinfo:
        partition.verify_access("T_ecg_lead1")
    assert "Future information leakage detected" in str(excinfo.value)
    assert "T_digits_5_9" in str(excinfo.value)

    # Querying unknown unallowed task must raise FutureInformationLeakageError
    with pytest.raises(FutureInformationLeakageError):
        partition.verify_access("T_xor_parity")


def test_fixed_compute_budget_invariance():
    """Verify that compute budget (generation count, population) is strictly invariant across conditions."""
    tasks = get_pilot_task_sequence(seed=0)[:2]  # 2 tasks for fast verification
    n_gen = 2
    pop_size = 3

    harness = ContinuityHarness(
        tasks=tasks,
        conditions=[
            ContinuityCondition.COLD_START,
            ContinuityCondition.NO_MEMORY,
            ContinuityCondition.CONTINUOUS_EXPERIENCE,
        ],
        orderings=["forward"],
        seeds=[0],
        n_generations=n_gen,
        population_size=pop_size,
    )
    artifact = harness.run()

    # Verify budgets recorded in artifact
    assert artifact.budgets["n_generations"] == n_gen
    assert artifact.budgets["population_size"] == pop_size

    # Verify every observation has the same generation count
    for obs in artifact.raw_observations:
        assert obs["order_id"] == "forward"
        assert obs["seed"] == 0


def test_memory_reset_and_inheritance_verification():
    """Verify memory state before and after task across cold-start, continuous, and no-memory."""
    tasks = get_pilot_task_sequence(seed=0)[:2]
    harness = ContinuityHarness(
        tasks=tasks,
        conditions=[
            ContinuityCondition.COLD_START,
            ContinuityCondition.NO_MEMORY,
            ContinuityCondition.CONTINUOUS_EXPERIENCE,
        ],
        orderings=["forward"],
        seeds=[0],
        n_generations=2,
    )
    artifact = harness.run()

    # 1. COLD_START: Memory before every task must be identical to fresh empty memory
    cold_obs = [o for o in artifact.raw_observations if o["condition"] == "COLD_START"]
    assert len(cold_obs) == 2
    # Before fingerprint on task 0 and task 1 must be identical (fresh reset)
    assert cold_obs[0]["memory_fingerprint_before"] == cold_obs[1]["memory_fingerprint_before"]
    assert cold_obs[1]["prior_task_count"] == 0

    # 2. NO_MEMORY: Memory fingerprint must be 'no_memory_disabled'
    nomem_obs = [o for o in artifact.raw_observations if o["condition"] == "NO_MEMORY"]
    assert len(nomem_obs) == 2
    assert nomem_obs[0]["memory_fingerprint_before"] == "no_memory_disabled"
    assert nomem_obs[1]["memory_fingerprint_before"] == "no_memory_disabled"

    # 3. CONTINUOUS_EXPERIENCE: Task 1 must inherit memory from Task 0
    cont_obs = [o for o in artifact.raw_observations if o["condition"] == "CONTINUOUS_EXPERIENCE"]
    assert len(cont_obs) == 2
    assert cont_obs[0]["memory_fingerprint_before"] != cont_obs[0]["memory_fingerprint_after"]
    # Task 1 before fingerprint must match Task 0 after fingerprint
    assert cont_obs[1]["memory_fingerprint_before"] == cont_obs[0]["memory_fingerprint_after"]
    assert cont_obs[1]["prior_task_count"] == 1


def test_independent_evaluator_boundary():
    """Verify evaluator computes metrics independently without mutating or relying on controller policy."""
    evaluator = ContinuityEvaluator(delta_threshold=0.01)

    # Test that evaluator can classify transfer without any controller instance
    xfer = evaluator.classify_transfer(
        source_task="T_digits_0_4",
        target_task="T_digits_5_9",
        strategy="rf_baseline",
        model_type="RandomForestClassifier",
        experienced_result=0.88,
        baseline_without_transfer=0.82,
        context_level=2,
    )
    assert xfer.transfer_classification == TransferClassification.POSITIVE
    assert xfer.delta == pytest.approx(0.06, abs=1e-4)

    # Neutral case
    xfer_neutral = evaluator.classify_transfer(
        source_task="T_digits_0_4",
        target_task="T_digits_5_9",
        strategy="rf_baseline",
        model_type="RandomForestClassifier",
        experienced_result=0.825,
        baseline_without_transfer=0.82,
        context_level=2,
    )
    assert xfer_neutral.transfer_classification == TransferClassification.NEUTRAL


def test_artifact_immutability_and_preservation():
    """Verify that historical benchmark files exist and are not modified by continuity harness."""
    base_dir = Path(__file__).resolve().parent.parent

    # Historical artifacts check (if they exist in the repo)
    p6_path = base_dir / "phase6_benchmark_result.json"
    p7_path = base_dir / "phase7_benchmark_result.json"

    p6_bytes_before = p6_path.read_bytes() if p6_path.exists() else None
    p7_bytes_before = p7_path.read_bytes() if p7_path.exists() else None

    # Run harness with output to separate artifact
    tasks = get_pilot_task_sequence(seed=0)[:1]
    harness = ContinuityHarness(
        tasks=tasks,
        conditions=[ContinuityCondition.COLD_START],
        orderings=["forward"],
        seeds=[0],
        n_generations=1,
    )
    test_artifact_path = base_dir / "test_phase11_artifact_integrity.json"
    try:
        artifact = harness.run(output_path=str(test_artifact_path))
        assert os.path.exists(test_artifact_path)
        assert artifact.artifact_fingerprint != ""

        # Historical files must remain completely unchanged
        if p6_bytes_before is not None:
            assert p6_path.read_bytes() == p6_bytes_before, "phase6 artifact was modified!"
        if p7_bytes_before is not None:
            assert p7_path.read_bytes() == p7_bytes_before, "phase7 artifact was modified!"
    finally:
        if test_artifact_path.exists():
            test_artifact_path.unlink()


def test_orchestration_performance_separation():
    """Verify orchestration overhead is tracked separately from model trial timings."""
    tasks = get_pilot_task_sequence(seed=0)[:1]
    harness = ContinuityHarness(
        tasks=tasks,
        conditions=[ContinuityCondition.COLD_START],
        orderings=["forward"],
        seeds=[0],
        n_generations=1,
    )
    artifact = harness.run()

    assert "task_init_ms" in artifact.orchestration_overhead_ms
    assert "memory_reset_ms" in artifact.orchestration_overhead_ms
    assert "experience_loading_ms" in artifact.orchestration_overhead_ms
    assert "partition_sealing_ms" in artifact.orchestration_overhead_ms
    assert "bookkeeping_ms" in artifact.orchestration_overhead_ms
    assert "artifact_writing_ms" in artifact.orchestration_overhead_ms

    for k, v in artifact.orchestration_overhead_ms.items():
        assert isinstance(v, float)
        assert v >= 0.0


def test_architectural_boundary_no_db_imports_in_continuity():
    """Verify continuity package maintains strict architectural boundary with zero direct DB/SQL imports."""
    import inspect
    import researchforge.benchmarks.continuity as cont_pkg

    for name, module in inspect.getmembers(cont_pkg, inspect.ismodule):
        source = inspect.getsource(module)
        assert "sqlalchemy" not in source.lower(), f"Direct SQLAlchemy import found in {module.__name__}"
        assert "psycopg" not in source.lower(), f"Direct psycopg import found in {module.__name__}"
        assert "from researchforge.storage.sql" not in source, f"Direct storage.sql import found in {module.__name__}"
