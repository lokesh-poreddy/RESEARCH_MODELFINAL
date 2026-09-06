import pytest
import dataclasses
import os
import json
import hashlib
import shutil
from typing import Dict, Any, Tuple, List
from pathlib import Path

from researchforge.benchmarks.execution.models import (
    ExecutionManifest, PreflightAttestation, LedgerEntry,
    RunManifestEntry, ExecutionStatus, DependencyPolicy,
    ExecutionCheckpoint, FrozenBenchmarkPackage
)
from researchforge.benchmarks.execution.engine import CheckpointManager
from researchforge.benchmarks.cohort.models import BenchmarkCondition, TaskFamily, TransferRegime
from researchforge.domain.base import _canonical_json, compute_canonical_fingerprint
from researchforge.benchmarks.execution.freeze import RawBenchmarkFreezer, FrozenBenchmarkError

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
        cohort_canonical_fingerprint=manifest.cohort_fingerprint,
        cohort_physical_file_sha256="cohort_physical_hash",
        sap_canonical_fingerprint=manifest.sap_fingerprint,
        manifest_canonical_fingerprint=manifest.manifest_fingerprint,
        manifest_physical_file_sha256="manifest_physical_hash",
        preflight_report_canonical_fingerprint="preflight_canonical",
        preflight_report_physical_file_sha256="preflight_physical",
        software_commit="abcd1234abcd1234abcd1234abcd1234abcd1234",
        dirty_worktree=False,
        preflight_pass=True,
        execution_authorized=True,
        authorized_by="approver",
        authorization_timestamp="2026-09-06T00:00:00Z"
    )
    # Recompute attestation_fingerprint
    d = pa.to_dict()
    d.pop("attestation_fingerprint")
    fp = compute_canonical_fingerprint(d)
    return dataclasses.replace(pa, attestation_fingerprint=fp)

def setup_valid_files(tmp_path: Path):
    manifest = create_mock_manifest()
    preflight = create_mock_preflight(manifest)
    
    cohort_p = tmp_path / "cohort.json"
    cohort_p.write_text('{"cohort": "test"}')
    c_hash = hashlib.sha256(cohort_p.read_bytes()).hexdigest()
    
    sap_p = tmp_path / "sap.json"
    sap_data = {"sap": "test"}
    sap_p.write_text(_canonical_json(sap_data) + "\n")
    s_hash = compute_canonical_fingerprint(sap_data)
    
    # Compute manifest with sap and cohort hashes
    manifest = dataclasses.replace(manifest,
                                   sap_fingerprint=s_hash,
                                   cohort_fingerprint=c_hash)
    manifest = dataclasses.replace(manifest, manifest_fingerprint=manifest.compute_fingerprint())
    
    # Now write manifest and get m_hash
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(_canonical_json(manifest.to_dict()) + "\n")
    m_hash = hashlib.sha256(manifest_p.read_bytes()).hexdigest()
    
    # Now update manifest fingerprint in manifest itself?
    # Wait, the manifest fingerprint is not stored in the manifest!
    # It's stored in PreflightAttestation and Checkpoint.
    
    pa = dataclasses.replace(preflight, 
        manifest_physical_file_sha256=m_hash,
        manifest_canonical_fingerprint=compute_canonical_fingerprint(manifest.to_dict()),
        cohort_physical_file_sha256=c_hash,
        sap_canonical_fingerprint=s_hash
    )
    
    d = pa.to_dict()
    d.pop("attestation_fingerprint")
    pa = dataclasses.replace(pa, attestation_fingerprint=compute_canonical_fingerprint(d))
    
    preflight_p = tmp_path / "preflight.json"
    preflight_p.write_text(_canonical_json(pa.to_dict()) + "\n")
    
    ledger_p = tmp_path / "ledger.jsonl"
    with open(ledger_p, "w") as f:
        for entry in manifest.entries:
            le = LedgerEntry(
                entry_id=entry.entry_id,
                sequence_group_id=entry.sequence_group_id,
                sequence_position=entry.sequence_position,
                condition=entry.condition.value,
                task_id=entry.task_id,
                seed=entry.seed,
                ordering=entry.ordering,
                execution_status=ExecutionStatus.COMPLETED.value,
                outcome_status="VALID_COMPLETED",
                best_metric=1.0,
                decision_quality=1.0,
                trials_executed=1,
                wallclock_seconds=1.0,
                memory_fingerprint_before="mem_before",
                memory_fingerprint_after="mem_after",
                entry_hash="hash",
                timestamp="2026-09-06T00:00:00Z"
            )
            f.write(_canonical_json(le.to_dict()) + "\n")
            
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    cm = CheckpointManager(ckpt_dir)
    for entry in manifest.entries:
        ckpt = ExecutionCheckpoint(
            entry_id=entry.entry_id,
            sequence_group_id=entry.sequence_group_id,
            sequence_position=entry.sequence_position,
            manifest_fingerprint=pa.manifest_canonical_fingerprint,
            cohort_fingerprint=pa.cohort_canonical_fingerprint,
            sap_fingerprint=pa.sap_canonical_fingerprint,
            software_commit=pa.software_commit,
            state_fingerprint="state_fp",
            parent_checkpoint_id=None,
            state_schema_version="1.0",
            serialized_state={"foo": "bar"}
        )
        cm.save(ckpt)
        
    return manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm, pa, manifest

def test_1_valid_freeze(tmp_path):
    manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm, preflight, manifest = setup_valid_files(tmp_path)
    
    out_dir = tmp_path / "out"
    pkg = RawBenchmarkFreezer.freeze(out_dir, manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm)
    
    assert pkg.package_version == "1.0.0"
    assert pkg.package_fingerprint  # non-empty fingerprint
    assert pkg.source_manifest_fingerprint == preflight.manifest_canonical_fingerprint
    
    frozen_root = out_dir / "artifacts" / "benchmarks" / "phase12b" / "frozen" / preflight.manifest_canonical_fingerprint
    assert frozen_root.exists()
    
    with open(frozen_root / "raw_benchmark_hashes.json", "r") as f:
        hashes = json.load(f)
    assert len(hashes) == 5
    
    # Check reproducible hash
    with open(frozen_root / "raw_benchmark_hashes.json", "r") as f:
        computed_pkg_hash = hashlib.sha256(_canonical_json(json.load(f)).encode("utf-8")).hexdigest()
    assert pkg.package_fingerprint == computed_pkg_hash


def test_2_missing_trials(tmp_path):
    manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm, preflight, manifest = setup_valid_files(tmp_path)
    
    # Remove last line of ledger
    lines = ledger_p.read_text().splitlines()
    ledger_p.write_text("\n".join(lines[:-1]) + "\n")
    
    out_dir = tmp_path / "out"
    with pytest.raises(FrozenBenchmarkError, match="Cannot freeze: Execution is incomplete"):
        RawBenchmarkFreezer.freeze(out_dir, manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm)
        
def test_3_modified_cohort_fails(tmp_path):
    manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm, preflight, manifest = setup_valid_files(tmp_path)
    
    cohort_p.write_text('{"cohort": "modified"}')
    
    out_dir = tmp_path / "out"
    with pytest.raises(FrozenBenchmarkError, match="Cohort physical fingerprint mutation detected."):
        RawBenchmarkFreezer.freeze(out_dir, manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm)

def test_4_modified_sap_fails(tmp_path):
    manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm, preflight, manifest = setup_valid_files(tmp_path)
    
    sap_p.write_text(_canonical_json({"sap": "modified"}) + "\n")
    
    out_dir = tmp_path / "out"
    with pytest.raises(FrozenBenchmarkError, match="SAP canonical fingerprint mutation detected."):
        RawBenchmarkFreezer.freeze(out_dir, manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm)

def test_5_modified_manifest_fails(tmp_path):
    manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm, preflight, manifest = setup_valid_files(tmp_path)
    
    manifest_p.write_text(manifest_p.read_text() + " ")
    
    out_dir = tmp_path / "out"
    with pytest.raises(FrozenBenchmarkError, match="Manifest physical fingerprint mutation detected."):
        RawBenchmarkFreezer.freeze(out_dir, manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm)

def test_6_modified_preflight_fails(tmp_path):
    manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm, preflight, manifest = setup_valid_files(tmp_path)
    
    data = json.loads(preflight_p.read_text())
    data["execution_authorized"] = False
    preflight_p.write_text(_canonical_json(data) + "\n")
    
    out_dir = tmp_path / "out"
    with pytest.raises(FrozenBenchmarkError, match="Preflight/authorization binding mutation detected."):
        RawBenchmarkFreezer.freeze(out_dir, manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm)

def test_7_missing_checkpoint(tmp_path):
    manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm, preflight, manifest = setup_valid_files(tmp_path)
    
    # delete one checkpoint
    os.remove(cm.checkpoints_dir / "task_2.json")
    
    out_dir = tmp_path / "out"
    # When a checkpoint is missing, CompletionAttestor records an artifact_gap,
    # which sets execution_complete=False. The freeze raises the generic
    # "Execution is incomplete" message (artifact_gaps > 0 causes this).
    with pytest.raises(FrozenBenchmarkError, match="Cannot freeze: Execution is incomplete"):
        RawBenchmarkFreezer.freeze(out_dir, manifest_p, preflight_p, cohort_p, sap_p, ledger_p, cm)

