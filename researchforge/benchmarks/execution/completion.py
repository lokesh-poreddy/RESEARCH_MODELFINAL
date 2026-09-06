import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional

from researchforge.domain.base import _canonical_json, compute_canonical_fingerprint
from researchforge.benchmarks.execution.models import (
    CompletionAttestation, ExecutionManifest, PreflightAttestation, 
    LedgerEntry, RunManifestEntry, ExecutionStatus, DependencyPolicy
)
from researchforge.benchmarks.execution.engine import CheckpointManager

class CompletionAttestor:
    
    @staticmethod
    def attest(
        manifest: ExecutionManifest,
        preflight: PreflightAttestation,
        ledger_path: Path | str,
        checkpoint_manager: CheckpointManager
    ) -> CompletionAttestation:
        
        # 1. Authorization & Lineage Verification
        if preflight.manifest_canonical_fingerprint != manifest.manifest_fingerprint:
            raise ValueError("Authorization chain mismatch: preflight manifest fingerprint != manifest fingerprint")
        if preflight.cohort_canonical_fingerprint != manifest.cohort_fingerprint:
            raise ValueError("Authorization chain mismatch: preflight cohort fingerprint != manifest cohort fingerprint")
        if preflight.sap_canonical_fingerprint != manifest.sap_fingerprint:
            raise ValueError("Authorization chain mismatch: preflight SAP fingerprint != manifest SAP fingerprint")
            
        # 2. Ledger Integrity
        path = Path(ledger_path)
        if not path.exists():
            physical_hash = hashlib.sha256(b"").hexdigest()
            lines = []
        else:
            with open(path, "rb") as f:
                content = f.read()
            physical_hash = hashlib.sha256(content).hexdigest()
            text = content.decode("utf-8")
            lines = [line for line in text.split('\n') if line]
            
        parsed_dicts = []
        invalid_records = 0
        invalid_entries = []
        ledger_integrity_passed = True
        
        for i, line in enumerate(lines):
            try:
                d = json.loads(line)
                expected_hash = d.get("entry_hash")
                temp = d.copy()
                temp.pop("entry_hash", None)
                computed = hashlib.sha256(_canonical_json(temp).encode("utf-8")).hexdigest()
                
                if expected_hash != computed:
                    invalid_records += 1
                    ledger_integrity_passed = False
                    eid = d.get("entry_id")
                    if eid:
                        invalid_entries.append(eid)
                else:
                    parsed_dicts.append(d)
                    
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    # Valid recoverable tail
                    pass
                else:
                    invalid_records += 1
                    ledger_integrity_passed = False
                    
        logical_list = [_canonical_json(d) for d in parsed_dicts]
        logical_ledger_fingerprint = hashlib.sha256(json.dumps(logical_list).encode("utf-8")).hexdigest()
        
        # 3. Identity & Field Integrity + 4. Accounting + 5. Causal Integrity
        expected_trials = len(manifest.entries)
        manifest_map = {e.entry_id: e for e in manifest.entries}
        
        seen_entries = set()
        seen_status: Dict[str, Tuple[str, str]] = {} # entry_id -> (exec_status, outcome_status)
        
        duplicates = 0
        unexpected = 0
        missing = 0
        non_completed = 0
        causal_violations = 0
        artifact_gaps = 0
        
        missing_entries = []
        duplicate_entries = []
        unexpected_entries = []
        non_completed_entries = []
        causal_violation_entries = []
        artifact_gap_entries = []
        
        highest_seq_pos_per_group = {}
        
        for d in parsed_dicts:
            entry_id = d.get("entry_id")
            if not entry_id:
                invalid_records += 1
                ledger_integrity_passed = False
                continue
                
            if entry_id not in manifest_map:
                unexpected += 1
                unexpected_entries.append(entry_id)
                continue
                
            if entry_id in seen_entries:
                duplicates += 1
                duplicate_entries.append(entry_id)
                continue
                
            m_entry = manifest_map[entry_id]
            
            # Verify immutable fields
            mismatch = False
            if d.get("sequence_group_id") != m_entry.sequence_group_id: mismatch = True
            if d.get("sequence_position") != m_entry.sequence_position: mismatch = True
            if d.get("condition") != m_entry.condition.value: mismatch = True
            if d.get("task_id") != m_entry.task_id: mismatch = True
            if d.get("seed") != m_entry.seed: mismatch = True
            if d.get("ordering") != m_entry.ordering: mismatch = True
            
            if mismatch:
                invalid_records += 1
                invalid_entries.append(entry_id)
                continue
                
            # Causal integrity
            is_causal_violation = False
            
            for dep in m_entry.depends_on_entry_ids:
                if dep not in seen_entries:
                    is_causal_violation = True
                    break
                dep_exec, dep_outcome = seen_status[dep]
                policy = m_entry.dependency_policy
                if policy == DependencyPolicy.EXECUTION_COMPLETE:
                    if dep_exec != ExecutionStatus.COMPLETED.value:
                        is_causal_violation = True
                        break
                elif policy == DependencyPolicy.SCIENTIFICALLY_VALID:
                    if dep_exec != ExecutionStatus.COMPLETED.value or dep_outcome != "VALID_COMPLETED":
                        is_causal_violation = True
                        break
                elif policy == DependencyPolicy.ARTIFACT_AVAILABLE:
                    # we only check if it completed
                    if dep_exec != ExecutionStatus.COMPLETED.value:
                        is_causal_violation = True
                        break
                        
            # Sequence position check
            group_id = m_entry.sequence_group_id
            pos = m_entry.sequence_position
            last_pos = highest_seq_pos_per_group.get(group_id, -1)
            if pos <= last_pos:
                is_causal_violation = True
            highest_seq_pos_per_group[group_id] = pos
            
            if is_causal_violation:
                causal_violations += 1
                causal_violation_entries.append(entry_id)
                
            # Status check
            exec_status = d.get("execution_status")
            outcome_status = d.get("outcome_status", "")
            
            if exec_status != ExecutionStatus.COMPLETED.value:
                non_completed += 1
                non_completed_entries.append(entry_id)
            else:
                # 6. Artifact Integrity
                if m_entry.condition.value not in ("COLD_START", "NO_MEMORY"):
                    ckpt = checkpoint_manager.load(entry_id)
                    if not ckpt:
                        artifact_gaps += 1
                        artifact_gap_entries.append(entry_id)
                    else:
                        try:
                            ckpt.validate_integrity(
                                manifest.manifest_fingerprint,
                                manifest.cohort_fingerprint,
                                manifest.sap_fingerprint,
                                preflight.software_commit
                            )
                            if (ckpt.entry_id != entry_id or 
                                ckpt.sequence_group_id != m_entry.sequence_group_id or 
                                ckpt.sequence_position != m_entry.sequence_position):
                                raise Exception("Identity mismatch")
                        except Exception:
                            artifact_gaps += 1
                            artifact_gap_entries.append(entry_id)
                            
            seen_entries.add(entry_id)
            seen_status[entry_id] = (exec_status, outcome_status)
            
        # 5. Missing
        for m_entry in manifest.entries:
            if m_entry.entry_id not in seen_entries:
                missing += 1
                missing_entries.append(m_entry.entry_id)
                
        # 7. Execution Complete
        execution_complete = (
            ledger_integrity_passed
            and missing == 0
            and duplicates == 0
            and unexpected == 0
            and causal_violations == 0
            and invalid_records == 0
            and artifact_gaps == 0
            and non_completed == 0
            and len(seen_entries) == expected_trials
        )
        
        # Build Attestation
        timestamp = datetime.now(timezone.utc).isoformat()
        
        temp_dict = {
            "manifest_fingerprint": manifest.manifest_fingerprint,
            "cohort_fingerprint": manifest.cohort_fingerprint,
            "sap_fingerprint": manifest.sap_fingerprint,
            "preflight_attestation_fingerprint": preflight.attestation_fingerprint,
            "software_commit": preflight.software_commit,
            
            "physical_ledger_fingerprint": physical_hash,
            "logical_ledger_fingerprint": logical_ledger_fingerprint,
            
            "expected_trials": expected_trials,
            "observed_unique_trials": len(seen_entries),
            "missing": missing,
            "duplicates": duplicates,
            "unexpected": unexpected,
            "non_completed": non_completed,
            "causal_violations": causal_violations,
            "invalid_records": invalid_records,
            "artifact_gaps": artifact_gaps,
            
            "ledger_integrity_passed": ledger_integrity_passed,
            "execution_complete": execution_complete,
            
            "missing_entries": missing_entries,
            "duplicate_entries": duplicate_entries,
            "unexpected_entries": unexpected_entries,
            "non_completed_entries": non_completed_entries,
            "causal_violation_entries": causal_violation_entries,
            "invalid_entries": invalid_entries,
            "artifact_gap_entries": artifact_gap_entries,
            
            "timestamp": timestamp,
        }
        
        attest_fp = compute_canonical_fingerprint(temp_dict)
        
        return CompletionAttestation(
            attestation_fingerprint=attest_fp,
            **temp_dict
        )
