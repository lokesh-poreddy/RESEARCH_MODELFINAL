import json
import os
import shutil
import tempfile
from pathlib import Path
import dataclasses
from unittest.mock import patch
import pytest

from researchforge.benchmarks.execution.models import (
    CheckpointInvalidError,
    DependencyPolicy,
    ExecutionCheckpoint,
    ExecutionManifest,
    ExecutionNotAuthorizedError,
    ExecutionStatus,
    LedgerEntry,
    ManifestInvalidError,
    PreflightAttestation,
    RunManifestEntry,
    get_software_commit_info,
)
from researchforge.benchmarks.execution.engine import BenchmarkExecutionEngine
from researchforge.benchmarks.cohort.models import BenchmarkCondition, TaskFamily, TransferRegime
from researchforge.domain.base import _canonical_json, compute_canonical_fingerprint


@pytest.fixture
def mock_commit():
    return "testcommit12345"

@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

@pytest.fixture
def setup_engine(workspace, mock_commit):
    ledger = workspace / "ledger.jsonl"
    registry = workspace / "registry.json"
    checkpoints = workspace / ".checkpoints"
    
    with patch("researchforge.benchmarks.execution.engine.get_software_commit_info", return_value=(mock_commit, False)):
        engine = BenchmarkExecutionEngine(ledger, registry, checkpoints)
        yield engine, workspace

def create_mock_manifest() -> ExecutionManifest:
    entry = RunManifestEntry(
        entry_id="task_1",
        sequence_group_id="group_1",
        sequence_position=1,
        condition=BenchmarkCondition.COLD_START,
        task_id="XOR_1",
        task_family=TaskFamily.DIGITS_SPATIAL,
        task_fingerprint="fp1",
        seed=42,
        ordering="default",
        prior_task_ids=(),
        prior_task_families=(TaskFamily.DIGITS_SPATIAL,),
        sequence_transfer_regime=TransferRegime.SAME_FAMILY,
        condition_effective_regime=TransferRegime.SAME_FAMILY,
        depends_on_entry_ids=(),
        dependency_policy=DependencyPolicy.EXECUTION_COMPLETE,
        generation_budget=1,
        population_size=2
    )
    return ExecutionManifest(
        entries=(entry,),
        cohort_fingerprint="cohort_fp",
        sap_fingerprint="sap_fp",
        manifest_fingerprint=""
    )

def finalize_manifest(m: ExecutionManifest) -> ExecutionManifest:
    fp = m.compute_fingerprint()
    return ExecutionManifest(
        entries=m.entries,
        cohort_fingerprint=m.cohort_fingerprint,
        sap_fingerprint=m.sap_fingerprint,
        manifest_fingerprint=fp
    )

def create_mock_attestation(m, preflight_pass=True, software_commit="testcommit12345"):
    a = PreflightAttestation(
        cohort_canonical_fingerprint=m.cohort_fingerprint,
        cohort_physical_file_sha256="c_phys",
        sap_canonical_fingerprint=m.sap_fingerprint,
        manifest_canonical_fingerprint=m.manifest_fingerprint,
        manifest_physical_file_sha256="m_phys",
        preflight_report_canonical_fingerprint="r_canon",
        preflight_report_physical_file_sha256="r_phys",
        software_commit=software_commit,
        dirty_worktree=False,
        preflight_pass=preflight_pass,
        execution_authorized=preflight_pass,
        authorized_by="admin" if preflight_pass else None,
        authorization_timestamp="now"
    )
    fp = a.compute_fingerprint()
    return dataclasses.replace(a, attestation_fingerprint=fp)

def create_valid_ledger_entry(entry_id="task_1", task_id="XOR_1", seq_pos=1):
    payload = {
        "timestamp": "now", "entry_id": entry_id, "sequence_group_id": "group_1", "sequence_position": seq_pos,
        "condition": "COLD_START", "task_id": task_id, "seed": 42, "ordering": "default",
        "execution_status": ExecutionStatus.COMPLETED.value, "outcome_status": "VALID_COMPLETED",
        "best_metric": 0.0, "decision_quality": 0.0, "trials_executed": 0, "wallclock_seconds": 0.0,
        "memory_fingerprint_before": "", "memory_fingerprint_after": ""
    }
    import hashlib
    entry_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return LedgerEntry(**{**payload, "entry_hash": entry_hash})


def test_1_unauthorized_execution(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, False, mock_commit)
    
    with pytest.raises(ExecutionNotAuthorizedError):
        engine.execute(manifest=m, attestation=a)
    
    assert not engine.ledger_path.exists()
    assert not list(engine.checkpoints.checkpoints_dir.iterdir() if engine.checkpoints.checkpoints_dir.exists() else [])

def test_2_invalid_attestation_fingerprint(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    a = dataclasses.replace(a, attestation_fingerprint="bad_fp")
    
    with pytest.raises(ExecutionNotAuthorizedError, match="cryptographic signature mismatch"):
        engine.execute(manifest=m, attestation=a)

def test_3_stale_software_commit(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, "old_commit")
    
    with pytest.raises(ExecutionNotAuthorizedError, match="Software commit SHA mismatch"):
        engine.execute(manifest=m, attestation=a)

def test_4_sealed_artifact_modification(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    # Modify manifest slightly
    m2 = ExecutionManifest(
        entries=m.entries,
        cohort_fingerprint="cohort_fp",
        sap_fingerprint="sap_fp",
        manifest_fingerprint="different"
    )
    a = create_mock_attestation(m2, True, mock_commit)
    
    with pytest.raises(ExecutionNotAuthorizedError, match="canonical fingerprint mismatch"):
        engine.execute(manifest=m2, attestation=a)

def test_5_missing_dependency(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = create_mock_manifest()
    entry1 = m.entries[0]
    entry2 = RunManifestEntry(
        entry_id="task_2",
        sequence_group_id="group_1",
        sequence_position=2,
        condition=BenchmarkCondition.CONTINUOUS_EXPERIENCE,
        task_id="XOR_2",
        task_family=TaskFamily.DIGITS_SPATIAL,
        task_fingerprint="fp2",
        seed=42,
        ordering="default",
        prior_task_ids=("task_1",),
        prior_task_families=(TaskFamily.DIGITS_SPATIAL,),
        sequence_transfer_regime=TransferRegime.SAME_FAMILY,
        condition_effective_regime=TransferRegime.SAME_FAMILY,
        depends_on_entry_ids=("task_1",),
        dependency_policy=DependencyPolicy.EXECUTION_COMPLETE,
        generation_budget=1,
        population_size=2
    )
    m = ExecutionManifest(entries=(entry1, entry2), cohort_fingerprint="c", sap_fingerprint="s", manifest_fingerprint="")
    m = finalize_manifest(m)
    a = create_mock_attestation(m, True, mock_commit)
    
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute") as mock_exec:
        # Mock executor to return a failed ledger entry for task_1
        mock_exec.return_value = (
            LedgerEntry(
                timestamp="now", entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
                condition="COLD_START", task_id="XOR_1", seed=42, ordering="default",
                execution_status=ExecutionStatus.FAILED.value, outcome_status="EXECUTION_FAILURE",
                best_metric=0.0, decision_quality=0.0, trials_executed=0, wallclock_seconds=0.0,
                memory_fingerprint_before="", memory_fingerprint_after="", entry_hash="hash"
            ),
            None
        )
        res = engine.execute(manifest=m, attestation=a, dry_run=False)
    
    # Should be 2 results: task_1 (FAILED), task_2 (BLOCKED)
    assert len(res) == 2
    assert res[0].execution_status == ExecutionStatus.FAILED.value
    assert res[1].execution_status == ExecutionStatus.BLOCKED.value

def test_14_manifest_cycle(setup_engine, mock_commit):
    engine, ws = setup_engine
    e1 = RunManifestEntry(
        entry_id="1", sequence_group_id="g1", sequence_position=1,
        condition=BenchmarkCondition.COLD_START, task_id="XOR", task_family=TaskFamily.DIGITS_SPATIAL, task_fingerprint="f", seed=42, ordering="d",
        prior_task_ids=(), prior_task_families=(), sequence_transfer_regime=TransferRegime.SAME_FAMILY,
        condition_effective_regime=TransferRegime.SAME_FAMILY, depends_on_entry_ids=("2",)
    )
    e2 = RunManifestEntry(
        entry_id="2", sequence_group_id="g1", sequence_position=2,
        condition=BenchmarkCondition.COLD_START, task_id="XOR", task_family=TaskFamily.DIGITS_SPATIAL, task_fingerprint="f", seed=42, ordering="d",
        prior_task_ids=(), prior_task_families=(), sequence_transfer_regime=TransferRegime.SAME_FAMILY,
        condition_effective_regime=TransferRegime.SAME_FAMILY, depends_on_entry_ids=("1",)
    )
    m = ExecutionManifest(entries=(e1, e2), cohort_fingerprint="c", sap_fingerprint="s", manifest_fingerprint="")
    m = finalize_manifest(m)
    a = create_mock_attestation(m, True, mock_commit)
    
    with pytest.raises(ManifestInvalidError, match="DAG Cycle"):
        engine.execute(m, a)

def test_16_dry_run_executes_zero(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    res = engine.execute(m, a, dry_run=True)
    assert len(res) == 0
    assert not engine.ledger_path.exists()

def test_6_dependency_policy(setup_engine, mock_commit):
    # Test EXECUTION_COMPLETE vs SCIENTIFICALLY_VALID vs ARTIFACT_AVAILABLE
    engine, ws = setup_engine
    m = create_mock_manifest()
    entry1 = m.entries[0]
    entry2 = dataclasses.replace(
        entry1,
        entry_id="task_2",
        sequence_position=2,
        depends_on_entry_ids=("task_1",),
        dependency_policy=DependencyPolicy.SCIENTIFICALLY_VALID
    )
    m = finalize_manifest(ExecutionManifest(entries=(entry1, entry2), cohort_fingerprint="c", sap_fingerprint="s", manifest_fingerprint=""))
    a = create_mock_attestation(m, True, mock_commit)
    
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute") as mock_exec:
        # Task 1 completes but has SCIENTIFIC_FAILURE
        mock_exec.return_value = (
            LedgerEntry(
                timestamp="now", entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
                condition="COLD_START", task_id="XOR_1", seed=42, ordering="default",
                execution_status=ExecutionStatus.COMPLETED.value, outcome_status="SCIENTIFIC_FAILURE",
                best_metric=0.0, decision_quality=0.0, trials_executed=0, wallclock_seconds=0.0,
                memory_fingerprint_before="", memory_fingerprint_after="", entry_hash="hash"
            ),
            None
        )
        res = engine.execute(manifest=m, attestation=a, dry_run=False)
    
    assert len(res) == 2
    assert res[0].execution_status == ExecutionStatus.COMPLETED.value
    # Since task_1 completed but failed scientifically, and task_2 requires SCIENTIFICALLY_VALID, task_2 should be BLOCKED
    assert res[1].execution_status == ExecutionStatus.BLOCKED.value


def test_7_duplicate_ledger_entry(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    # Pre-populate ledger with duplicate entries
    le = create_valid_ledger_entry("task_1", "XOR_1", 1)
    engine.writer.append(le)
    engine.writer.append(le) # Duplicate
    
    with pytest.raises(Exception, match="Duplicate ledger entry|Integrity"):
        engine.execute(manifest=m, attestation=a)

def test_9_corrupted_checkpoint(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    ckpt = ExecutionCheckpoint(
        entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
        manifest_fingerprint=m.manifest_fingerprint, cohort_fingerprint="c", sap_fingerprint="s",
        software_commit=mock_commit, state_fingerprint="bad", parent_checkpoint_id=None,
        state_schema_version="1", serialized_state={}
    )
    engine.checkpoints.save(ckpt)
    
    le = create_valid_ledger_entry("task_1", "XOR_1", 1)
    engine.writer.append(le)
    
    with pytest.raises(CheckpointInvalidError):
        engine.execute(manifest=m, attestation=a)


def test_11_interrupted_run_resumable(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = create_mock_manifest()
    entry1 = m.entries[0]
    entry2 = dataclasses.replace(
        entry1,
        entry_id="task_2", sequence_position=2, depends_on_entry_ids=("task_1",)
    )
    m = finalize_manifest(ExecutionManifest(entries=(entry1, entry2), cohort_fingerprint="c", sap_fingerprint="s", manifest_fingerprint=""))
    a = create_mock_attestation(m, True, mock_commit)
    
    # Pre-populate task_1 completed
    ckpt = ExecutionCheckpoint(
        entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
        manifest_fingerprint=m.manifest_fingerprint, cohort_fingerprint=m.cohort_fingerprint, sap_fingerprint=m.sap_fingerprint,
        software_commit=mock_commit, state_fingerprint=compute_canonical_fingerprint({}), parent_checkpoint_id=None,
        state_schema_version="1.0.0", serialized_state={}
    )
    engine.checkpoints.save(ckpt)
    
    le = create_valid_ledger_entry("task_1", "XOR_1", 1)
    engine.writer.append(le)
    
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute") as mock_exec:
        mock_exec.return_value = (
            LedgerEntry(
                timestamp="now", entry_id="task_2", sequence_group_id="group_1", sequence_position=2,
                condition="COLD_START", task_id="XOR_1", seed=42, ordering="default",
                execution_status=ExecutionStatus.COMPLETED.value, outcome_status="VALID_COMPLETED",
                best_metric=0.0, decision_quality=0.0, trials_executed=0, wallclock_seconds=0.0,
                memory_fingerprint_before="", memory_fingerprint_after="", entry_hash="hash"
            ),
            None
        )
        res = engine.execute(manifest=m, attestation=a, dry_run=False)
    
    assert len(res) == 1
    assert res[0].entry_id == "task_2"


def test_12_completed_trial_never_rerun(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    ckpt = ExecutionCheckpoint(
        entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
        manifest_fingerprint=m.manifest_fingerprint, cohort_fingerprint=m.cohort_fingerprint, sap_fingerprint=m.sap_fingerprint,
        software_commit=mock_commit, state_fingerprint=compute_canonical_fingerprint({}), parent_checkpoint_id=None,
        state_schema_version="1.0.0", serialized_state={}
    )
    engine.checkpoints.save(ckpt)
    
    le = create_valid_ledger_entry("task_1", "XOR_1", 1)
    engine.writer.append(le)
    
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute") as mock_exec:
        res = engine.execute(manifest=m, attestation=a, dry_run=False)
        mock_exec.assert_not_called()
    assert len(res) == 0

def test_15_invalid_sequence_ordering(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = create_mock_manifest()
    entry1 = m.entries[0]
    entry2 = dataclasses.replace(
        entry1,
        entry_id="task_2", sequence_position=1,  # Duplicate sequence_position in same group!
        depends_on_entry_ids=("task_1",)
    )
    m = finalize_manifest(ExecutionManifest(entries=(entry1, entry2), cohort_fingerprint="c", sap_fingerprint="s", manifest_fingerprint=""))
    a = create_mock_attestation(m, True, mock_commit)
    
    with pytest.raises(ManifestInvalidError, match="Invalid sequence positions"):
        engine.execute(manifest=m, attestation=a)


def test_8_corrupted_ledger_record(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    le = LedgerEntry(
        timestamp="now", entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
        condition="COLD_START", task_id="XOR_1", seed=42, ordering="default",
        execution_status=ExecutionStatus.COMPLETED.value, outcome_status="VALID_COMPLETED",
        best_metric=0.0, decision_quality=0.0, trials_executed=0, wallclock_seconds=0.0,
        memory_fingerprint_before="", memory_fingerprint_after="", entry_hash="hash"
    )
    # Write manually and corrupt it
    engine.writer.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine.writer.ledger_path, "w") as f:
        d = le.to_dict()
        d["entry_hash"] = "invalid_hash"
        f.write(_canonical_json(d) + "\n")
        
    with pytest.raises(Exception, match="Corrupted|Integrity|hash"):
        engine.execute(manifest=m, attestation=a)

def test_10_checkpoint_wrong_manifest(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    ckpt = ExecutionCheckpoint(
        entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
        manifest_fingerprint="WRONG_MANIFEST_FINGERPRINT", cohort_fingerprint=m.cohort_fingerprint, sap_fingerprint=m.sap_fingerprint,
        software_commit=mock_commit, state_fingerprint=compute_canonical_fingerprint({}), parent_checkpoint_id=None,
        state_schema_version="1.0.0", serialized_state={}
    )
    engine.checkpoints.save(ckpt)
    
    le = create_valid_ledger_entry("task_1", "XOR_1", 1)
    engine.writer.append(le)
    
    with pytest.raises(CheckpointInvalidError, match="mismatch"):
        engine.execute(manifest=m, attestation=a)


def test_13_scientific_failure_execution_failure(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute") as mock_exec:
        mock_exec.return_value = (
            LedgerEntry(
                timestamp="now", entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
                condition="COLD_START", task_id="XOR_1", seed=42, ordering="default",
                execution_status=ExecutionStatus.COMPLETED.value, outcome_status="SCIENTIFIC_FAILURE",
                best_metric=0.0, decision_quality=0.0, trials_executed=0, wallclock_seconds=0.0,
                memory_fingerprint_before="", memory_fingerprint_after="", entry_hash="hash"
            ),
            None
        )
        res = engine.execute(manifest=m, attestation=a, dry_run=False)
        
    assert len(res) == 1
    assert res[0].execution_status == ExecutionStatus.COMPLETED.value
    assert res[0].outcome_status == "SCIENTIFIC_FAILURE"

def test_20_repeated_resume_idempotent(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = create_mock_manifest()
    entry1 = m.entries[0]
    entry2 = dataclasses.replace(
        entry1,
        entry_id="task_2", sequence_position=2, depends_on_entry_ids=("task_1",)
    )
    entry3 = dataclasses.replace(
        entry1,
        entry_id="task_3", sequence_position=3, depends_on_entry_ids=("task_2",)
    )
    m = finalize_manifest(ExecutionManifest(entries=(entry1, entry2, entry3), cohort_fingerprint="c", sap_fingerprint="s", manifest_fingerprint=""))
    a = create_mock_attestation(m, True, mock_commit)
    
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute") as mock_exec:
        # First run: task 1, 2, 3 complete
        mock_exec.side_effect = [
            (create_valid_ledger_entry("task_1", "XOR_1", 1), {}),
            (create_valid_ledger_entry("task_2", "XOR_1", 2), {}),
            (create_valid_ledger_entry("task_3", "XOR_1", 3), {})
        ]
        res1 = engine.execute(manifest=m, attestation=a)
        
        # Second run: everything should be skipped (no mock_exec calls)
        res2 = engine.execute(manifest=m, attestation=a)
        
        # Third run: skipped again
        res3 = engine.execute(manifest=m, attestation=a)
        
    assert len(res1) == 3
    assert len(res2) == 0
    assert len(res3) == 0
    
    # Read ledger lines
    with open(engine.ledger_path, "r") as f:
        lines = f.readlines()
        
    assert len(lines) == 3
    assert json.loads(lines[0])["entry_id"] == "task_1"
    assert json.loads(lines[1])["entry_id"] == "task_2"
    assert json.loads(lines[2])["entry_id"] == "task_3"

