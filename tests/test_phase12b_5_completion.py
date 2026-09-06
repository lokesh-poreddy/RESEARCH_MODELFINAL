import pytest
import dataclasses
import os
import json
import hashlib
from typing import Dict, Any, Tuple, List
from pathlib import Path

from researchforge.benchmarks.execution.completion import CompletionAttestor
from researchforge.benchmarks.execution.models import (
    ExecutionManifest, PreflightAttestation, LedgerEntry,
    RunManifestEntry, ExecutionStatus, DependencyPolicy,
    ExecutionCheckpoint
)
from researchforge.benchmarks.execution.engine import CheckpointManager
from researchforge.benchmarks.cohort.models import BenchmarkCondition, TaskFamily, TransferRegime
from researchforge.domain.base import _canonical_json, compute_canonical_fingerprint

def finalize_manifest(manifest: ExecutionManifest) -> ExecutionManifest:
    fp = compute_canonical_fingerprint({
        "entries": [e.to_dict() for e in manifest.entries],
        "cohort_fingerprint": manifest.cohort_fingerprint,
        "sap_fingerprint": manifest.sap_fingerprint
    })
    return dataclasses.replace(manifest, manifest_fingerprint=fp)

def create_mock_manifest(num_entries=3) -> ExecutionManifest:
    entries = []
    for i in range(1, num_entries + 1):
        deps = () if i == 1 else (f"task_{i-1}",)
        entry = RunManifestEntry(
            entry_id=f"task_{i}",
            sequence_group_id="group_1",
            sequence_position=i,
            condition=BenchmarkCondition.CONTINUOUS_EXPERIENCE,
            task_id="XOR_1",
            task_family=TaskFamily.DIGITS_SPATIAL,
            task_fingerprint="fp1",
            seed=42,
            ordering="default",
            depends_on_entry_ids=deps,
            dependency_policy=DependencyPolicy.SCIENTIFICALLY_VALID,
            prior_task_ids=(),
            prior_task_families=(),
            sequence_transfer_regime=TransferRegime.NO_PRIOR_EXPERIENCE,
            condition_effective_regime=TransferRegime.NO_PRIOR_EXPERIENCE
        )
        entries.append(entry)
    
    m = ExecutionManifest(
        entries=tuple(entries),
        cohort_fingerprint="cohort_fp",
        sap_fingerprint="sap_fp",
        manifest_fingerprint=""
    )
    return finalize_manifest(m)

def create_mock_preflight(manifest: ExecutionManifest) -> PreflightAttestation:
    pa = PreflightAttestation(
        attestation_fingerprint="",
        manifest_canonical_fingerprint=manifest.manifest_fingerprint,
        cohort_canonical_fingerprint=manifest.cohort_fingerprint,
        sap_canonical_fingerprint=manifest.sap_fingerprint,
        software_commit={"commit": "abc", "dirty": False},
        execution_authorized=True,
        authorized_by="test",
        authorization_timestamp="now"
    )
    fp = compute_canonical_fingerprint({
        "manifest_canonical_fingerprint": pa.manifest_canonical_fingerprint,
        "cohort_canonical_fingerprint": pa.cohort_canonical_fingerprint,
        "sap_canonical_fingerprint": pa.sap_canonical_fingerprint,
        "software_commit": pa.software_commit,
        "execution_authorized": pa.execution_authorized,
        "authorized_by": pa.authorized_by,
        "authorization_timestamp": pa.authorization_timestamp
    })
    return dataclasses.replace(pa, attestation_fingerprint=fp)

def create_ledger_entry(entry_id, seq_pos, status="COMPLETED", outcome="VALID_COMPLETED") -> LedgerEntry:
    payload = {
        "timestamp": "now", "entry_id": entry_id, "sequence_group_id": "group_1", "sequence_position": seq_pos,
        "condition": "CONTINUOUS_EXPERIENCE", "task_id": "XOR_1", "seed": 42, "ordering": "default",
        "execution_status": status, "outcome_status": outcome,
        "best_metric": 0.0, "decision_quality": 0.0, "trials_executed": 0, "wallclock_seconds": 0.0,
        "memory_fingerprint_before": "", "memory_fingerprint_after": ""
    }
    h = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return LedgerEntry(**payload, entry_hash=h)

def create_checkpoint(entry_id, seq_pos, manifest, preflight) -> ExecutionCheckpoint:
    state = {}
    ser_state = state
    state_fp = compute_canonical_fingerprint(ser_state)
    return ExecutionCheckpoint(
        entry_id=entry_id,
        sequence_group_id="group_1",
        sequence_position=seq_pos,
        manifest_fingerprint=manifest.manifest_fingerprint,
        cohort_fingerprint=manifest.cohort_fingerprint,
        sap_fingerprint=manifest.sap_fingerprint,
        software_commit=preflight.software_commit,
        state_fingerprint=state_fp,
        parent_checkpoint_id=None,
        state_schema_version="1.0.0",
        serialized_state=ser_state
    )

@pytest.fixture
def workspace(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ckpt_mgr = CheckpointManager(tmp_path / "checkpoints")
    return ledger_path, ckpt_mgr

def write_ledger(path, entries: List[LedgerEntry]):
    with open(path, "w") as f:
        for e in entries:
            f.write(_canonical_json(e.to_dict()) + "\n")

# 1. Valid complete execution
def test_1_valid_completion(workspace):
    path, ckpt = workspace
    m = create_mock_manifest()
    p = create_mock_preflight(m)
    
    entries = []
    for i in range(1, 4):
        eid = f"task_{i}"
        entries.append(create_ledger_entry(eid, i))
        ckpt.save(create_checkpoint(eid, i, m, p))
        
    write_ledger(path, entries)
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.execution_complete is True
    assert att.missing == 0
    assert att.invalid_records == 0
    assert att.causal_violations == 0

# 2. Missing trial
def test_2_missing_trial(workspace):
    path, ckpt = workspace
    m = create_mock_manifest()
    p = create_mock_preflight(m)
    
    # Only write task_1
    entries = [create_ledger_entry("task_1", 1)]
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    write_ledger(path, entries)
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.execution_complete is False
    assert att.missing == 2
    assert att.missing_entries == ["task_2", "task_3"]

# 3. Duplicate trial
def test_3_duplicate_trial(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    # Write task_1 twice
    e = create_ledger_entry("task_1", 1)
    entries = [e, e]
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    write_ledger(path, entries)
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.execution_complete is False
    assert att.duplicates == 1
    assert "task_1" in att.duplicate_entries

# 4. Unexpected trial
def test_4_unexpected_trial(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    e2 = create_ledger_entry("task_99", 99)
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    write_ledger(path, [e1, e2])
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.unexpected == 1
    assert "task_99" in att.unexpected_entries

# 5. Malformed JSON
def test_5_malformed_json(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    write_ledger(path, [e1])
    
    with open(path, "a") as f:
        f.write("{broken_json: \n")
        
    att = CompletionAttestor.attest(m, p, path, ckpt)
    # the last line is trailing partial, so it's ignored! Wait, if it's the last line, it's ignored.
    # to test malformed json counting as invalid, it must be NON-final.
    
    with open(path, "a") as f:
        f.write('{"timestamp": "now", "entry_hash": "a"}\n') # valid json, but wrong hash?
        
    # let's rewrite the file to have broken JSON in the middle
    with open(path, "w") as f:
        f.write("{broken json}\n")
        f.write(_canonical_json(e1.to_dict()) + "\n")
        
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.invalid_records == 1
    assert att.ledger_integrity_passed is False

# 6. Tampered entry hash
def test_6_tampered_entry_hash(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    d = e1.to_dict()
    d["entry_hash"] = "tampered"
    
    with open(path, "w") as f:
        f.write(_canonical_json(d) + "\n")
        
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.invalid_records == 1
    assert att.ledger_integrity_passed is False
    assert "task_1" in att.invalid_entries

# 7. Trailing partial JSON
def test_7_trailing_partial_json(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    write_ledger(path, [e1])
    
    with open(path, "a") as f:
        f.write('{"timestamp": "now", "ent')
        
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.ledger_integrity_passed is True
    assert att.invalid_records == 0
    assert att.execution_complete is True

# 8. Non-final corrupted JSON
def test_8_non_final_corrupted(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(2)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    e2 = create_ledger_entry("task_2", 2)
    with open(path, "w") as f:
        f.write(_canonical_json(e1.to_dict()) + "\n")
        f.write('{"broken\n')
        f.write(_canonical_json(e2.to_dict()) + "\n")
        
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.invalid_records == 1
    assert att.ledger_integrity_passed is False

# 9. Dependency order violation
def test_9_dependency_order_violation(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(2)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    e2 = create_ledger_entry("task_2", 2)
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    ckpt.save(create_checkpoint("task_2", 2, m, p))
    
    # write task_2 BEFORE task_1
    write_ledger(path, [e2, e1])
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.causal_violations == 2
    assert "task_2" in att.causal_violation_entries
    assert "task_1" in att.causal_violation_entries

# 10. Sequence order violation
def test_10_sequence_order_violation(workspace):
    path, ckpt = workspace
    # make task 1 and 2 independent but in same group
    m = create_mock_manifest(2)
    # remove dependency from task 2 so it doesn't fail dependency check
    entries = list(m.entries)
    entries[1] = dataclasses.replace(entries[1], depends_on_entry_ids=())
    m = finalize_manifest(dataclasses.replace(m, entries=tuple(entries)))
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    e2 = create_ledger_entry("task_2", 2)
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    ckpt.save(create_checkpoint("task_2", 2, m, p))
    
    # write task_2 BEFORE task_1. Sequence pos 2 then 1 violates monotonic order.
    write_ledger(path, [e2, e1])
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.causal_violations == 1
    assert "task_1" in att.causal_violation_entries

# 11. Dependency policy violation
def test_11_dependency_policy_violation(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(2) # task_2 requires SCIENTIFICALLY_VALID task_1
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1, status="COMPLETED", outcome="SCIENTIFIC_FAILURE")
    e2 = create_ledger_entry("task_2", 2)
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    ckpt.save(create_checkpoint("task_2", 2, m, p))
    write_ledger(path, [e1, e2])
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.causal_violations == 1
    assert "task_2" in att.causal_violation_entries

# 12. Manifest field mismatch
def test_12_manifest_field_mismatch(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    d = e1.to_dict()
    d["seed"] = 999 # mismatch with manifest seed 42
    
    # re-hash to make ledger integrity pass, so it's a valid record physically, but invalid logically vs manifest
    temp = d.copy()
    temp.pop("entry_hash", None)
    d["entry_hash"] = hashlib.sha256(_canonical_json(temp).encode("utf-8")).hexdigest()
    
    with open(path, "w") as f:
        f.write(_canonical_json(d) + "\n")
        
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.invalid_records == 1
    assert "task_1" in att.invalid_entries
    assert att.ledger_integrity_passed is True

# 13. Missing checkpoint
def test_13_missing_checkpoint(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    write_ledger(path, [e1])
    # no checkpoint saved
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.artifact_gaps == 1
    assert "task_1" in att.artifact_gap_entries

# 14. Tampered checkpoint
def test_14_tampered_checkpoint(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    c = create_checkpoint("task_1", 1, m, p)
    # tamper state fingerprint
    c = dataclasses.replace(c, state_fingerprint="tampered")
    ckpt.save(c)
    write_ledger(path, [e1])
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.artifact_gaps == 1

# 15 to 19 Checkpoint mismatched properties
@pytest.mark.parametrize("kwarg", [
    {"manifest_fingerprint": "wrong"},
    {"cohort_fingerprint": "wrong"},
    {"sap_fingerprint": "wrong"},
    {"software_commit": {"commit": "wrong", "dirty": True}},
    {"entry_id": "wrong"},
])
def test_15_to_19_checkpoint_mismatches(workspace, kwarg):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    c = create_checkpoint("task_1", 1, m, p)
    c = dataclasses.replace(c, **kwarg)
    ckpt.save(c)
    write_ledger(path, [e1])
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.artifact_gaps == 1

# 20. BLOCKED entry is non_completed, not missing
def test_20_blocked_entry(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1, status="BLOCKED")
    write_ledger(path, [e1])
    # no checkpoint needed for non-completed
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.non_completed == 1
    assert att.missing == 0
    assert "task_1" in att.non_completed_entries
    assert att.execution_complete is False

# 21. SCIENTIFIC_FAILURE is execution complete
def test_21_scientific_failure_complete(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1, status="COMPLETED", outcome="SCIENTIFIC_FAILURE")
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    write_ledger(path, [e1])
    
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.execution_complete is True

# 22. Authorization-chain mismatch
def test_22_authorization_mismatch(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    p = dataclasses.replace(p, manifest_canonical_fingerprint="tampered")
    
    with pytest.raises(ValueError, match="Authorization chain mismatch"):
        CompletionAttestor.attest(m, p, path, ckpt)

# 23. Physical vs logical ledger fingerprint uniqueness
def test_23_fingerprint_uniqueness(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    
    # Write normal
    write_ledger(path, [e1])
    att1 = CompletionAttestor.attest(m, p, path, ckpt)
    
    # Write with trailing partial
    with open(path, "a") as f:
        f.write('{"trail\n')
        
    att2 = CompletionAttestor.attest(m, p, path, ckpt)
    
    assert att1.physical_ledger_fingerprint != att2.physical_ledger_fingerprint
    assert att1.logical_ledger_fingerprint == att2.logical_ledger_fingerprint

# 24. Repeated attestation is deterministic
def test_24_deterministic(workspace):
    import datetime
    from unittest.mock import patch
    path, ckpt = workspace
    m = create_mock_manifest(1)
    p = create_mock_preflight(m)
    e1 = create_ledger_entry("task_1", 1)
    ckpt.save(create_checkpoint("task_1", 1, m, p))
    write_ledger(path, [e1])
    
    with patch("researchforge.benchmarks.execution.completion.datetime") as mock_dt:
        mock_dt.now.return_value = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        att1 = CompletionAttestor.attest(m, p, path, ckpt)
        att2 = CompletionAttestor.attest(m, p, path, ckpt)
    
    assert att1.attestation_fingerprint == att2.attestation_fingerprint
    
# 25. Checkpoints handle COLD_START correctly
def test_25_cold_start_no_checkpoint_needed(workspace):
    path, ckpt = workspace
    m = create_mock_manifest(1)
    # change to COLD_START
    entries = list(m.entries)
    entries[0] = dataclasses.replace(entries[0], condition=BenchmarkCondition.COLD_START)
    m = finalize_manifest(dataclasses.replace(m, entries=tuple(entries)))
    p = create_mock_preflight(m)
    
    e1 = create_ledger_entry("task_1", 1)
    d = e1.to_dict()
    d["condition"] = "COLD_START"
    temp = d.copy()
    temp.pop("entry_hash", None)
    d["entry_hash"] = hashlib.sha256(_canonical_json(temp).encode("utf-8")).hexdigest()
    
    with open(path, "w") as f:
        f.write(_canonical_json(d) + "\n")
        
    # DO NOT create checkpoint
    att = CompletionAttestor.attest(m, p, path, ckpt)
    assert att.artifact_gaps == 0
    assert att.execution_complete is True
