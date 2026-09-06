import json
import os
import shutil
import tempfile
from pathlib import Path
import dataclasses
from unittest.mock import patch, MagicMock
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
from researchforge.benchmarks.execution.engine import (
    BenchmarkExecutionEngine,
    LedgerRecoveryReader,
    ReconciliationState,
    ReconciliationError,
    ExecutionReconciler
)
from researchforge.benchmarks.cohort.models import BenchmarkCondition, TaskFamily, TransferRegime
from researchforge.domain.base import _canonical_json, compute_canonical_fingerprint

# Helper functions for tests
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
        condition=BenchmarkCondition.CONTINUOUS_EXPERIENCE,
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
        "condition": "CONTINUOUS_EXPERIENCE", "task_id": task_id, "seed": 42, "ordering": "default",
        "execution_status": ExecutionStatus.COMPLETED.value, "outcome_status": "VALID_COMPLETED",
        "best_metric": 0.0, "decision_quality": 0.0, "trials_executed": 0, "wallclock_seconds": 0.0,
        "memory_fingerprint_before": "", "memory_fingerprint_after": ""
    }
    import hashlib
    entry_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return LedgerEntry(**{**payload, "entry_hash": entry_hash})


# Test 1: trailing partial JSON -> safely recoverable
def test_1_trailing_partial_json(setup_engine):
    engine, ws = setup_engine
    le = create_valid_ledger_entry("task_1")
    engine.writer.append(le)
    
    with open(engine.ledger_path, "a", encoding="utf-8") as f:
        f.write('{"timestamp": "now", "entry_id": "task_') # incomplete
        
    entries, state = LedgerRecoveryReader.read_ledger(engine.ledger_path)
    assert state == ReconciliationState.RECOVERABLE
    assert "task_1" in entries

# Test 2: corrupted non-final ledger line -> fatal
def test_2_corrupted_nonfinal_line(setup_engine):
    engine, ws = setup_engine
    with open(engine.ledger_path, "w", encoding="utf-8") as f:
        f.write('{"corrupted": \n')
        f.write(_canonical_json(create_valid_ledger_entry("task_1").to_dict()) + "\n")
        
    with pytest.raises(ReconciliationError, match="LEDGER_CORRUPTION"):
        LedgerRecoveryReader.read_ledger(engine.ledger_path)

# Test 3: atomic checkpoint replacement verified
def test_3_atomic_checkpoint_replacement(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    ckpt = ExecutionCheckpoint(
        entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
        manifest_fingerprint=m.manifest_fingerprint, cohort_fingerprint=m.cohort_fingerprint,
        sap_fingerprint=m.sap_fingerprint, software_commit=mock_commit,
        state_fingerprint=compute_canonical_fingerprint({}), parent_checkpoint_id=None,
        state_schema_version="1", serialized_state={}
    )
    
    with patch("os.replace") as mock_replace:
        engine.checkpoints.save(ckpt)
        assert mock_replace.called
        args = mock_replace.call_args[0]
        assert str(args[0]).endswith(".tmp")
        assert str(args[1]).endswith(".json")

# Test 4: missing checkpoint -> fatal
def test_4_missing_checkpoint(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    engine.writer.append(create_valid_ledger_entry("task_1"))
    
    with pytest.raises(ReconciliationError, match="Missing checkpoint"):
        engine.execute(m, a)

# Test 5: tampered checkpoint -> fatal
def test_5_tampered_checkpoint(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    ckpt = ExecutionCheckpoint(
        entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
        manifest_fingerprint=m.manifest_fingerprint, cohort_fingerprint=m.cohort_fingerprint,
        sap_fingerprint=m.sap_fingerprint, software_commit=mock_commit,
        state_fingerprint="bad", parent_checkpoint_id=None,
        state_schema_version="1", serialized_state={}
    )
    engine.checkpoints.save(ckpt)
    engine.writer.append(create_valid_ledger_entry("task_1"))
    
    with pytest.raises(ReconciliationError, match="Tampered checkpoint"):
        engine.execute(m, a)

# Test 6: manifest mismatch -> fatal
def test_6_manifest_mismatch(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    ckpt = ExecutionCheckpoint(
        entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
        manifest_fingerprint="wrong", cohort_fingerprint=m.cohort_fingerprint,
        sap_fingerprint=m.sap_fingerprint, software_commit=mock_commit,
        state_fingerprint=compute_canonical_fingerprint({}), parent_checkpoint_id=None,
        state_schema_version="1", serialized_state={}
    )
    engine.checkpoints.save(ckpt)
    engine.writer.append(create_valid_ledger_entry("task_1"))
    
    with pytest.raises(ReconciliationError, match="Tampered checkpoint.*manifest"):
        engine.execute(m, a)

# Test 7, 8, 9, 10: Other mismatches
def test_7_to_10_other_mismatches(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    # Cohort mismatch
    ckpt = ExecutionCheckpoint(
        entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
        manifest_fingerprint=m.manifest_fingerprint, cohort_fingerprint="wrong",
        sap_fingerprint=m.sap_fingerprint, software_commit=mock_commit,
        state_fingerprint=compute_canonical_fingerprint({}), parent_checkpoint_id=None,
        state_schema_version="1", serialized_state={}
    )
    engine.checkpoints.save(ckpt)
    engine.writer.append(create_valid_ledger_entry("task_1"))
    with pytest.raises(ReconciliationError, match="Tampered checkpoint.*cohort"):
        engine.execute(m, a)
        
    # Software commit mismatch
    ckpt = dataclasses.replace(ckpt, cohort_fingerprint=m.cohort_fingerprint, software_commit="wrong")
    engine.checkpoints.save(ckpt)
    with pytest.raises(ReconciliationError, match="Tampered checkpoint.*commit"):
        engine.execute(m, a)

# Test 11: perfectly synchronized ledger/checkpoint -> success
def test_11_synchronized_success(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    ckpt = ExecutionCheckpoint(
        entry_id="task_1", sequence_group_id="group_1", sequence_position=1,
        manifest_fingerprint=m.manifest_fingerprint, cohort_fingerprint=m.cohort_fingerprint,
        sap_fingerprint=m.sap_fingerprint, software_commit=mock_commit,
        state_fingerprint=compute_canonical_fingerprint({}), parent_checkpoint_id=None,
        state_schema_version="1", serialized_state={}
    )
    engine.checkpoints.save(ckpt)
    engine.writer.append(create_valid_ledger_entry("task_1"))
    
    # Should not raise any error, and return empty list since task_1 is done
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute") as mock_exec:
        res = engine.execute(m, a)
        assert len(res) == 0

# Test 12, 13, 14: Execution flow and crash simulation
def test_12_execution_ordering(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    order = []
    
    def mock_execute(*args, **kwargs):
        order.append("exec")
        return create_valid_ledger_entry("task_1"), {}
        
    def mock_save(*args, **kwargs):
        order.append("save_ckpt")
        
    def mock_append(*args, **kwargs):
        order.append("append_ledger")
        
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute", side_effect=mock_execute), \
         patch("researchforge.benchmarks.execution.engine.CheckpointManager.save", side_effect=mock_save), \
         patch("researchforge.benchmarks.execution.engine.ExecutionLedgerWriter.append", side_effect=mock_append):
        engine.execute(m, a)
        
    assert order == ["exec", "save_ckpt", "append_ledger"]

# Test 13: Case A - crash before checkpoint
def test_13_case_a_crash_before_checkpoint(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    def mock_execute(*args, **kwargs):
        return create_valid_ledger_entry("task_1"), {}
        
    def mock_save(*args, **kwargs):
        raise KeyboardInterrupt("Simulated crash")
        
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute", side_effect=mock_execute), \
         patch("researchforge.benchmarks.execution.engine.CheckpointManager.save", side_effect=mock_save):
        with pytest.raises(KeyboardInterrupt):
            engine.execute(m, a)
            
    assert not engine.ledger_path.exists()
    
# Test 14: Case B - checkpoint durable, crash before ledger
def test_14_case_b_orphan_checkpoint(setup_engine, mock_commit):
    engine, ws = setup_engine
    m = finalize_manifest(create_mock_manifest())
    a = create_mock_attestation(m, True, mock_commit)
    
    def mock_execute(*args, **kwargs):
        return create_valid_ledger_entry("task_1"), {}
        
    def mock_append(*args, **kwargs):
        raise KeyboardInterrupt("Simulated crash")
        
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute", side_effect=mock_execute), \
         patch("researchforge.benchmarks.execution.engine.ExecutionLedgerWriter.append", side_effect=mock_append):
        with pytest.raises(KeyboardInterrupt):
            engine.execute(m, a)
            
    # Checkpoint should exist, but ledger empty
    assert engine.checkpoints.load("task_1") is not None
    assert not engine.ledger_path.exists()
    
    # On resume, because ledger doesn't have it, we should re-run task_1
    with patch("researchforge.benchmarks.execution.engine.TrialExecutor.execute") as mock_exec:
        mock_exec.return_value = (create_valid_ledger_entry("task_1"), {})
        res = engine.execute(m, a)
        assert len(res) == 1
        assert mock_exec.called

# Test 19: temporary checkpoints ignored
def test_19_temporary_checkpoints_ignored(setup_engine, mock_commit):
    engine, ws = setup_engine
    tmp_path = engine.checkpoints.checkpoints_dir / "task_1.json.tmp"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        f.write('{"broken": true}')
        
    # load() should return None for task_1 since .json doesn't exist
    assert engine.checkpoints.load("task_1") is None
