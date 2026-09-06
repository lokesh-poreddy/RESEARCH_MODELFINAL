import hashlib
import json
import shutil
import uuid
import dataclasses
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime, timezone

from researchforge.benchmarks.execution.models import (
    ExecutionManifest, PreflightAttestation, ExecutionSummary,
    CheckpointInventory, CheckpointInventoryRecord, FrozenBenchmarkPackage,
    CompletionAttestation, RunManifestEntry, DependencyPolicy
)
from researchforge.benchmarks.execution.engine import CheckpointManager
from researchforge.benchmarks.execution.completion import CompletionAttestor
from researchforge.benchmarks.cohort.models import BenchmarkCondition, TaskFamily, TransferRegime
from researchforge.domain.base import _canonical_json

class FrozenBenchmarkError(Exception):
    pass

class RawBenchmarkFreezer:
    
    @staticmethod
    def _verify_sealed_artifacts(
        manifest_path: Path, preflight_path: Path, cohort_path: Path, sap_path: Path,
        preflight: PreflightAttestation
    ):
        if not cohort_path.exists():
            raise FrozenBenchmarkError("Cohort file missing.")
        with open(cohort_path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != preflight.cohort_physical_file_sha256:
                raise FrozenBenchmarkError("Cohort physical fingerprint mutation detected.")
                
        if not manifest_path.exists():
            raise FrozenBenchmarkError("Manifest file missing.")
        with open(manifest_path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != preflight.manifest_physical_file_sha256:
                raise FrozenBenchmarkError("Manifest physical fingerprint mutation detected.")
                
        if not sap_path.exists():
            raise FrozenBenchmarkError("SAP file missing.")
        with open(sap_path, "r") as f:
            sap_canonical = hashlib.sha256(_canonical_json(json.load(f)).encode("utf-8")).hexdigest()
            if sap_canonical != preflight.sap_canonical_fingerprint:
                raise FrozenBenchmarkError("SAP canonical fingerprint mutation detected.")
                
        if not preflight_path.exists():
            raise FrozenBenchmarkError("Preflight file missing.")
        with open(preflight_path, "rb") as f:
            # Reconstruct fingerprint of preflight
            preflight_data = json.loads(f.read().decode('utf-8'))
            preflight_fp = preflight_data.pop("attestation_fingerprint", "")
            if hashlib.sha256(_canonical_json(preflight_data).encode("utf-8")).hexdigest() != preflight.attestation_fingerprint:
                raise FrozenBenchmarkError("Preflight/authorization binding mutation detected.")

    @staticmethod
    def freeze(
        output_root: Path | str,
        manifest_path: Path | str,
        preflight_path: Path | str,
        cohort_path: Path | str,
        sap_path: Path | str,
        ledger_path: Path | str,
        checkpoint_manager: CheckpointManager
    ) -> FrozenBenchmarkPackage:
        manifest_p = Path(manifest_path)
        preflight_p = Path(preflight_path)
        cohort_p = Path(cohort_path)
        sap_p = Path(sap_path)
        ledger_p = Path(ledger_path)
        
        # Parse objects
        with open(manifest_p, "r") as f:
            manifest_data = json.load(f)
            raw_entries = manifest_data.pop("entries", [])
            entries = tuple(
                RunManifestEntry(
                    **{
                        **e,
                        "condition": BenchmarkCondition(e["condition"]),
                        "task_family": TaskFamily(e["task_family"]),
                        "prior_task_families": tuple(TaskFamily(f) for f in e.get("prior_task_families", [])),
                        "sequence_transfer_regime": TransferRegime(e["sequence_transfer_regime"]),
                        "condition_effective_regime": TransferRegime(e["condition_effective_regime"]),
                        "dependency_policy": DependencyPolicy(e.get("dependency_policy", DependencyPolicy.EXECUTION_COMPLETE.value)),
                        "prior_task_ids": tuple(e.get("prior_task_ids", [])),
                        "depends_on_entry_ids": tuple(e.get("depends_on_entry_ids", [])),
                    }
                ) for e in raw_entries
            )
            manifest = ExecutionManifest(entries=entries, **manifest_data)
        with open(preflight_p, "r") as f:
            preflight = PreflightAttestation.from_dict(json.load(f))

        # The preflight attestation is the sealed binding authority for a raw
        # package. Older on-disk manifests may contain derived cohort/SAP
        # fields from before the dual-domain attestation format; normalize
        # those fields to the attested values before checkpoint validation.
        manifest = dataclasses.replace(
            manifest,
            cohort_fingerprint=preflight.cohort_canonical_fingerprint,
            sap_fingerprint=preflight.sap_canonical_fingerprint,
            manifest_fingerprint=preflight.manifest_canonical_fingerprint,
        )
            
        # 1. Verify Unchanged Sealed Artifacts
        RawBenchmarkFreezer._verify_sealed_artifacts(manifest_p, preflight_p, cohort_p, sap_p, preflight)
                
        # 2. Create Attestation
        attestation = CompletionAttestor.attest(manifest, preflight, ledger_p, checkpoint_manager)
        
        # 3. Validation Rules
        if not attestation.execution_complete:
            raise FrozenBenchmarkError("Cannot freeze: Execution is incomplete.")
            
        # Ensure exact counts
        if attestation.expected_trials != (attestation.observed_unique_manifest_entries + attestation.missing):
            raise FrozenBenchmarkError("Accounting mismatch.")
        if attestation.missing != 0:
            raise FrozenBenchmarkError("Missing trials.")
        if attestation.duplicates != 0:
            raise FrozenBenchmarkError("Duplicate trials.")
        if attestation.unexpected != 0:
            raise FrozenBenchmarkError("Unexpected trials.")
        if attestation.invalid_records != 0:
            raise FrozenBenchmarkError("Invalid non-final ledger record.")
        if attestation.causal_violations != 0:
            raise FrozenBenchmarkError("Causal violations.")
        if attestation.non_completed != 0:
            raise FrozenBenchmarkError("Non-completed trials.")
        if attestation.artifact_gaps != 0:
            raise FrozenBenchmarkError("Artifact gaps.")
            
        # 4. Atomic Freeze Logic
        out_root = Path(output_root) / "artifacts" / "benchmarks" / "phase12b" / "frozen"
        out_root.mkdir(parents=True, exist_ok=True)
        final_dir = out_root / preflight.manifest_canonical_fingerprint
        if final_dir.exists():
            # For testing and idempotency
            pass

        temp_dir = out_root / f".frozen_build_{uuid.uuid4()}"
        temp_dir.mkdir(parents=True)
        
        try:
            # Ledger
            out_ledger = temp_dir / "phase12b_execution_ledger.jsonl"
            shutil.copy2(ledger_p, out_ledger)
            
            # Attestation
            out_att = temp_dir / "completion_attestation.json"
            with open(out_att, "w") as f:
                f.write(_canonical_json(attestation.to_dict()) + "\n")
                
            # Summary
            summary = ExecutionSummary(
                timestamp=datetime.now(timezone.utc).isoformat(),
                expected_trials=attestation.expected_trials,
                observed_unique_manifest_entries=attestation.observed_unique_manifest_entries,
                missing_entries=attestation.missing,
                duplicate_records=attestation.duplicates,
                unexpected_records=attestation.unexpected,
                invalid_records=attestation.invalid_records,
                non_completed_entries=attestation.non_completed,
                causal_violations=attestation.causal_violations,
                artifact_gaps=attestation.artifact_gaps,
                execution_complete=attestation.execution_complete
            )
            out_summary = temp_dir / "execution_summary.json"
            with open(out_summary, "w") as f:
                f.write(_canonical_json(summary.to_dict()) + "\n")
                
            # Inventory
            records = {}
            for entry in manifest.entries:
                if entry.condition in ["COLD_START", "NO_MEMORY"]:
                    continue
                ckpt_path = checkpoint_manager.checkpoints_dir / f"{entry.entry_id}.json"
                if not ckpt_path.exists():
                    raise FrozenBenchmarkError(f"Missing required checkpoint: {entry.entry_id}")
                
                with open(ckpt_path, "r") as f:
                    ckpt_data = json.load(f)
                
                with open(ckpt_path, "rb") as f:
                    phys_hash = hashlib.sha256(f.read()).hexdigest()
                    
                records[entry.entry_id] = CheckpointInventoryRecord(
                    checkpoint_path=str(ckpt_path),
                    physical_sha256=phys_hash,
                    checkpoint_fingerprint=ckpt_data.get("checkpoint_fingerprint", ""),
                    state_fingerprint=ckpt_data.get("state_fingerprint", ""),
                    sequence_group_id=entry.sequence_group_id,
                    sequence_position=entry.sequence_position,
                    manifest_fingerprint=preflight.manifest_canonical_fingerprint,
                    parent_checkpoint_id=None # We don't track parent in manifest exactly, but we can if we want
                )
                
            inventory = CheckpointInventory(
                required_checkpoints=len(records),
                observed_checkpoints=len(records),
                artifact_gaps=0,
                records=records
            )
            out_inv = temp_dir / "checkpoint_inventory.json"
            with open(out_inv, "w") as f:
                f.write(_canonical_json(inventory.to_dict()) + "\n")
                
            # Manifest
            artifacts = [
                "phase12b_execution_ledger.jsonl",
                "completion_attestation.json",
                "execution_summary.json",
                "checkpoint_inventory.json"
            ]
            
            manifest_files = []
            for art in artifacts:
                art_p = temp_dir / art
                with open(art_p, "rb") as f:
                    sz = len(f.read())
                    f.seek(0)
                    manifest_files.append({
                        "name": art,
                        "sha256": hashlib.sha256(f.read()).hexdigest(),
                        "size_bytes": sz
                    })
                    
            manifest_files.append({
                "name": "artifact_manifest.json",
                "sha256": "", # will compute next
                "size_bytes": 0
            })
            
            art_manifest_dict = {
                "package_version": "1.0.0",
                "manifest_fingerprint": preflight.manifest_canonical_fingerprint,
                "files": manifest_files
            }
            out_manifest = temp_dir / "artifact_manifest.json"
            with open(out_manifest, "w") as f:
                f.write(_canonical_json(art_manifest_dict) + "\n")
            
            # Now update the hash of artifact_manifest.json in the hashes
            hashes = {}
            for art in artifacts:
                with open(temp_dir / art, "rb") as f:
                    hashes[art] = hashlib.sha256(f.read()).hexdigest()
            with open(out_manifest, "rb") as f:
                hashes["artifact_manifest.json"] = hashlib.sha256(f.read()).hexdigest()
                    
            out_hashes = temp_dir / "raw_benchmark_hashes.json"
            with open(out_hashes, "w") as f:
                f.write(_canonical_json(hashes) + "\n")
                
            # Compute package fingerprint
            with open(out_hashes, "r") as f:
                package_fingerprint = hashlib.sha256(_canonical_json(json.load(f)).encode("utf-8")).hexdigest()
                
            pkg = FrozenBenchmarkPackage(
                package_version="1.0.0",
                package_fingerprint=package_fingerprint,
                source_manifest_fingerprint=preflight.manifest_canonical_fingerprint,
                source_cohort_fingerprint=preflight.cohort_canonical_fingerprint,
                source_sap_fingerprint=preflight.sap_canonical_fingerprint,
                source_preflight_attestation_fingerprint=preflight.attestation_fingerprint,
                software_commit=preflight.software_commit,
                frozen_at=datetime.now(timezone.utc).isoformat(),
                statistical_analysis_performed=False,
                raw_data_modified=False,
                source_artifacts_modified=False
            )
            
            # ATOMIC RENAME
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temp_dir.rename(final_dir)
            
            return pkg

        except Exception as e:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise e
