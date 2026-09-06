"""researchforge/benchmarks/execution/engine.py — Phase 12B.3 Authorized Benchmark Execution Engine.

This module implements the strict execution orchestration layer for the 32,400-trial benchmark.
It enforces complete cryptographic authorization, DAG dependencies, topological validity, 
ledger integrity, and canonical restart-safe checkpointing.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...domain.base import (
    _canonical_json,
    compute_canonical_fingerprint,
    compute_file_sha256,
)
from ...pipeline.controller import ResearchController
from ..cohort.models import BenchmarkCondition, RunOutcomeCategory
from ..continuity.evaluator import ContinuityEvaluator
from .materialization import materialize_benchmark_task
from .models import (
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


class AuthorizationVerifier:
    """Verifies complete cryptographic binding before execution."""
    @staticmethod
    def verify(
        manifest: ExecutionManifest,
        attestation: PreflightAttestation,
        manifest_file_path: Optional[Path | str],
        cohort_file_path: Optional[Path | str],
        sap_file_path: Optional[Path | str]
    ) -> None:
        if not attestation.execution_authorized:
            raise ExecutionNotAuthorizedError("Human authorization is FALSE.")
        
        recomputed_fp = attestation.compute_fingerprint()
        if attestation.attestation_fingerprint and attestation.attestation_fingerprint != recomputed_fp:
            raise ExecutionNotAuthorizedError("Attestation cryptographic signature mismatch.")
        
        # Verify Manifest
        recomputed_manifest_fp = manifest.compute_fingerprint()
        if manifest.manifest_fingerprint and manifest.manifest_fingerprint != recomputed_manifest_fp:
            raise ExecutionNotAuthorizedError("Manifest internal canonical fingerprint mismatch.")
        if attestation.manifest_canonical_fingerprint != manifest.manifest_fingerprint:
            raise ExecutionNotAuthorizedError("Attestation manifest fingerprint does not match loaded manifest.")
        if manifest_file_path and Path(manifest_file_path).exists():
            phys = compute_file_sha256(manifest_file_path)
            if attestation.manifest_physical_file_sha256 and phys != attestation.manifest_physical_file_sha256:
                raise ExecutionNotAuthorizedError("Manifest physical file SHA-256 mismatch.")
        
        # Verify Cohort
        if attestation.cohort_canonical_fingerprint != manifest.cohort_fingerprint:
            raise ExecutionNotAuthorizedError("Attestation cohort fingerprint does not match manifest cohort fingerprint.")
        if cohort_file_path and Path(cohort_file_path).exists():
            phys = compute_file_sha256(cohort_file_path)
            if attestation.cohort_physical_file_sha256 and phys != attestation.cohort_physical_file_sha256:
                raise ExecutionNotAuthorizedError("Cohort physical file SHA-256 mismatch.")
                
        # Verify SAP
        # The attestation has sap_canonical_fingerprint
        if sap_file_path and Path(sap_file_path).exists():
            with open(sap_file_path, "r", encoding="utf-8") as f:
                sap_data = json.load(f)
            sap_fp = compute_canonical_fingerprint(sap_data)
            if attestation.sap_canonical_fingerprint != sap_fp:
                raise ExecutionNotAuthorizedError("SAP canonical fingerprint mismatch.")
                
        # Verify Git Commit
        current_git_sha, _ = get_software_commit_info()
        if attestation.software_commit != current_git_sha:
            raise ExecutionNotAuthorizedError("Software commit SHA mismatch.")


class ManifestValidator:
    """Performs strict DAG validation of the execution manifest."""
    @staticmethod
    def validate(manifest: ExecutionManifest) -> None:
        entries = manifest.entries
        entry_ids = set()
        
        for e in entries:
            if e.entry_id in entry_ids:
                raise ManifestInvalidError(f"Duplicate entry_id: {e.entry_id}")
            entry_ids.add(e.entry_id)
        
        for e in entries:
            for dep in e.depends_on_entry_ids:
                if dep not in entry_ids:
                    raise ManifestInvalidError(f"Dependency {dep} for {e.entry_id} not found in manifest.")
        
        # Cycle detection (DFS)
        adj = {e.entry_id: list(e.depends_on_entry_ids) for e in entries}
        visited = set()
        rec_stack = set()
        
        def is_cyclic(v: str) -> bool:
            visited.add(v)
            rec_stack.add(v)
            for neighbor in adj.get(v, []):
                if neighbor not in visited:
                    if is_cyclic(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(v)
            return False
            
        for node in adj:
            if node not in visited:
                if is_cyclic(node):
                    raise ManifestInvalidError(f"DAG Cycle detected involving {node}.")
                    
        # Verify sequence position uniqueness within groups
        groups = {}
        for e in entries:
            groups.setdefault(e.sequence_group_id, set())
            if e.sequence_position in groups[e.sequence_group_id]:
                raise ManifestInvalidError(f"Invalid sequence positions: duplicate position {e.sequence_position} in group {e.sequence_group_id}")
            groups[e.sequence_group_id].add(e.sequence_position)


class ExecutionLedgerWriter:
    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path

    def append(self, entry: LedgerEntry) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(_canonical_json(entry.to_dict()) + "\n")


class LedgerIntegrityValidator:
    """Ensures ledger entries are fully complete and valid."""
    @staticmethod
    def is_resumably_complete(entry_dict: Dict[str, Any]) -> bool:
        if entry_dict.get("execution_status") != ExecutionStatus.COMPLETED.value:
            return False
        
        recorded_hash = entry_dict.get("entry_hash", "")
        if not recorded_hash:
            return False
        
        temp_dict = dict(entry_dict)
        temp_dict.pop("entry_hash", None)
        recomputed = hashlib.sha256(_canonical_json(temp_dict).encode("utf-8")).hexdigest()
        return recomputed == recorded_hash


class CheckpointManager:
    def __init__(self, checkpoints_dir: Path):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, checkpoint: ExecutionCheckpoint) -> None:
        path = self.checkpoints_dir / f"{checkpoint.entry_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, sort_keys=True)
            
    def load(self, entry_id: str) -> Optional[ExecutionCheckpoint]:
        path = self.checkpoints_dir / f"{entry_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExecutionCheckpoint(**data)


class DependencyScheduler:
    @staticmethod
    def is_ready(
        entry: RunManifestEntry,
        completed_entries: Dict[str, Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        for dep_id in entry.depends_on_entry_ids:
            if dep_id not in completed_entries:
                return False, f"Dependency {dep_id} missing or incomplete."
            
            dep_record = completed_entries[dep_id]
            
            if entry.dependency_policy == DependencyPolicy.SCIENTIFICALLY_VALID.value:
                if dep_record.get("outcome_status") != RunOutcomeCategory.VALID_COMPLETED.value:
                    return False, f"Dependency {dep_id} did not yield a valid scientific outcome."
                    
            elif entry.dependency_policy == DependencyPolicy.EXECUTION_COMPLETE.value:
                if dep_record.get("execution_status") != ExecutionStatus.COMPLETED.value:
                    return False, f"Dependency {dep_id} did not complete execution."
        
        return True, None


class TrialExecutor:
    def __init__(self):
        self.evaluator = ContinuityEvaluator()
        
    def execute(
        self,
        entry: RunManifestEntry,
        inherited_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[LedgerEntry, Dict[str, Any]]:
        t_start = time.time()

        m_task = materialize_benchmark_task(entry.task_id, seed=entry.seed)

        cond = entry.condition
        if cond == BenchmarkCondition.COLD_START:
            ctrl = ResearchController(
                m_task.task,
                condition="cold_start",
                seed=entry.seed,
                population_size=entry.population_size,
            )
        elif cond == BenchmarkCondition.NO_MEMORY:
            ctrl = ResearchController(
                m_task.task,
                condition="no_memory",
                seed=entry.seed,
                population_size=entry.population_size,
            )
        else:
            state = inherited_state or {}
            ctrl = ResearchController(
                m_task.task,
                condition=cond.value.lower(),
                seed=entry.seed,
                population_size=entry.population_size,
                ecrm=state.get("ecrm"),
                trajectory_memory=state.get("trajectory_memory"),
                adaptive_trajectory_memory=state.get("adaptive_trajectory_memory"),
                policy_learner=state.get("policy_learner"),
                failed_signatures=state.get("failed_signatures") or set(),
            )

        mem_before_fp = ctrl.export_memory_fingerprint()
        
        try:
            run_res = ctrl.run(n_generations=entry.generation_budget)
            fails = inherited_state.get("failed_signatures", set()) if inherited_state else set()
            dq = self.evaluator.calculate_decision_quality(run_res.trials, fails)
            outcome_status = RunOutcomeCategory.VALID_COMPLETED.value if run_res.best_metric > 0.0 else RunOutcomeCategory.SCIENTIFIC_FAILURE.value
            execution_status = ExecutionStatus.COMPLETED.value
            best_metric = run_res.best_metric
            trials_executed = len(run_res.trials)
        except Exception:
            outcome_status = "EXECUTION_FAILURE"
            execution_status = ExecutionStatus.FAILED.value
            dq = 0.0
            best_metric = 0.0
            trials_executed = 0

        mem_after_fp = ctrl.export_memory_fingerprint()
        wallclock = time.time() - t_start

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entry_id": entry.entry_id,
            "sequence_group_id": entry.sequence_group_id,
            "sequence_position": entry.sequence_position,
            "condition": entry.condition.value,
            "task_id": entry.task_id,
            "seed": entry.seed,
            "ordering": entry.ordering,
            "execution_status": execution_status,
            "outcome_status": outcome_status,
            "best_metric": best_metric,
            "decision_quality": dq,
            "trials_executed": trials_executed,
            "wallclock_seconds": wallclock,
            "memory_fingerprint_before": mem_before_fp,
            "memory_fingerprint_after": mem_after_fp,
        }
        entry_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

        ledger_entry = LedgerEntry(
            timestamp=payload["timestamp"],
            entry_id=entry.entry_id,
            sequence_group_id=entry.sequence_group_id,
            sequence_position=entry.sequence_position,
            condition=entry.condition.value,
            task_id=entry.task_id,
            seed=entry.seed,
            ordering=entry.ordering,
            execution_status=execution_status,
            outcome_status=outcome_status,
            best_metric=best_metric,
            decision_quality=dq,
            trials_executed=trials_executed,
            wallclock_seconds=wallclock,
            memory_fingerprint_before=mem_before_fp,
            memory_fingerprint_after=mem_after_fp,
            entry_hash=entry_hash,
        )

        next_state = {
            "ecrm": ctrl.ecrm,
            "trajectory_memory": ctrl.trajectory_memory,
            "adaptive_trajectory_memory": ctrl.adaptive_trajectory_memory,
            "policy_learner": ctrl.policy_learner,
            "failed_signatures": set(ctrl._failed_signatures),
            "memory_fingerprint": mem_after_fp,
        }

        return ledger_entry, next_state


class BenchmarkExecutionEngine:
    def __init__(
        self,
        ledger_path: Path,
        registry_path: Path,
        checkpoints_dir: Path
    ):
        self.ledger_path = Path(ledger_path)
        self.registry_path = Path(registry_path)
        self.writer = ExecutionLedgerWriter(self.ledger_path)
        self.checkpoints = CheckpointManager(Path(checkpoints_dir))
        self.executor = TrialExecutor()

    def load_completed_entries(self) -> Dict[str, Dict[str, Any]]:
        completed = {}
        if not self.ledger_path.exists():
            return completed
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("execution_status") == ExecutionStatus.COMPLETED.value:
                    if not LedgerIntegrityValidator.is_resumably_complete(record):
                        raise ValueError(f"Corrupted ledger record detected for entry_id: {record.get('entry_id')}")
                    if record["entry_id"] in completed:
                        raise ValueError(f"Duplicate ledger entry detected for entry_id: {record['entry_id']}")
                    completed[record["entry_id"]] = record
        return completed

    def _state_to_serializable(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "failed_signatures": list(state.get("failed_signatures", set())),
            "memory_fingerprint": state.get("memory_fingerprint", ""),
            "ecrm_records": [r.__dict__ for r in state["ecrm"].records] if state.get("ecrm") else [],
            "traj_records": [r.__dict__ for r in state["trajectory_memory"].records] if state.get("trajectory_memory") else [],
            "adapt_records": [r.__dict__ for r in state["adaptive_trajectory_memory"].records] if state.get("adaptive_trajectory_memory") else [],
            "policy_q": state["policy_learner"].q if state.get("policy_learner") else {},
            "policy_times_tried": state["policy_learner"].times_tried if state.get("policy_learner") else {},
            "policy_total_trials": state["policy_learner"].total_trials if state.get("policy_learner") else 0,
        }
        
    def _serializable_to_state(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        from ...memory.ecrm import ECRM, MemoryRecord
        from ...memory.trajectory import TrajectoryMemory, TrajectoryRecord
        from ...memory.adaptive_trajectory import AdaptiveTrajectoryMemory, AdaptiveTrajectoryRecord
        from ...policy.policy_learner import PolicyLearner
        from ...genome.operators import STRATEGIES

        state = {}
        state["failed_signatures"] = set(tuple(x) for x in serialized.get("failed_signatures", []))
        state["memory_fingerprint"] = serialized.get("memory_fingerprint", "")
        
        ecrm = ECRM()
        ecrm.records = [MemoryRecord(**r) for r in serialized.get("ecrm_records", [])]
        state["ecrm"] = ecrm
        
        tm = TrajectoryMemory()
        tm.records = [TrajectoryRecord(**r) for r in serialized.get("traj_records", [])]
        state["trajectory_memory"] = tm
        
        atm = AdaptiveTrajectoryMemory()
        atm.records = [AdaptiveTrajectoryRecord(**r) for r in serialized.get("adapt_records", [])]
        state["adaptive_trajectory_memory"] = atm
        
        policy = PolicyLearner(STRATEGIES)
        policy.q = serialized.get("policy_q", {s: 0.0 for s in STRATEGIES})
        policy.times_tried = serialized.get("policy_times_tried", {s: 0 for s in STRATEGIES})
        policy.total_trials = serialized.get("policy_total_trials", 0)
        state["policy_learner"] = policy
        
        return state

    def execute(
        self,
        manifest: ExecutionManifest,
        attestation: PreflightAttestation,
        manifest_file_path: Optional[Path | str] = None,
        cohort_file_path: Optional[Path | str] = None,
        sap_file_path: Optional[Path | str] = None,
        dry_run: bool = False
    ) -> List[LedgerEntry]:
        AuthorizationVerifier.verify(
            manifest=manifest,
            attestation=attestation,
            manifest_file_path=manifest_file_path,
            cohort_file_path=cohort_file_path,
            sap_file_path=sap_file_path
        )
        
        ManifestValidator.validate(manifest)
        completed_entries = self.load_completed_entries()
        
        groups: Dict[str, List[RunManifestEntry]] = {}
        for e in manifest.entries:
            groups.setdefault(e.sequence_group_id, []).append(e)
            
        results: List[LedgerEntry] = []
        
        for gid, entries in groups.items():
            entries.sort(key=lambda x: x.sequence_position)
            current_state: Optional[Dict[str, Any]] = None
            parent_checkpoint_id: Optional[str] = None
            
            for e in entries:
                if e.entry_id in completed_entries:
                    if not dry_run:
                        ckpt = self.checkpoints.load(e.entry_id)
                        if ckpt:
                            ckpt.validate_integrity(
                                manifest.manifest_fingerprint,
                                manifest.cohort_fingerprint,
                                manifest.sap_fingerprint,
                                attestation.software_commit
                            )
                            current_state = self._serializable_to_state(ckpt.serialized_state)
                            parent_checkpoint_id = e.entry_id
                    continue
                
                ready, reason = DependencyScheduler.is_ready(e, completed_entries)
                if not ready:
                    if not dry_run:
                        payload = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "entry_id": e.entry_id,
                            "sequence_group_id": e.sequence_group_id,
                            "sequence_position": e.sequence_position,
                            "condition": e.condition.value,
                            "task_id": e.task_id,
                            "seed": e.seed,
                            "ordering": e.ordering,
                            "execution_status": ExecutionStatus.BLOCKED.value,
                            "outcome_status": "INVALID_RUN",
                            "best_metric": 0.0,
                            "decision_quality": 0.0,
                            "trials_executed": 0,
                            "wallclock_seconds": 0.0,
                            "memory_fingerprint_before": "",
                            "memory_fingerprint_after": "",
                        }
                        entry_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
                        le = LedgerEntry(**{**payload, "entry_hash": entry_hash})
                        self.writer.append(le)
                        results.append(le)
                    continue
                
                if dry_run:
                    completed_entries[e.entry_id] = {
                        "execution_status": ExecutionStatus.COMPLETED.value,
                        "outcome_status": RunOutcomeCategory.VALID_COMPLETED.value
                    }
                    continue
                    
                ledger_record, next_state = self.executor.execute(e, current_state)
                results.append(ledger_record)
                self.writer.append(ledger_record)
                
                if ledger_record.execution_status == ExecutionStatus.COMPLETED.value:
                    completed_entries[e.entry_id] = ledger_record.to_dict()
                    if e.condition not in (BenchmarkCondition.COLD_START, BenchmarkCondition.NO_MEMORY):
                        current_state = next_state
                        
                        ser_state = self._state_to_serializable(current_state)
                        state_fp = compute_canonical_fingerprint(ser_state)
                        ckpt = ExecutionCheckpoint(
                            entry_id=e.entry_id,
                            sequence_group_id=e.sequence_group_id,
                            sequence_position=e.sequence_position,
                            manifest_fingerprint=manifest.manifest_fingerprint,
                            cohort_fingerprint=manifest.cohort_fingerprint,
                            sap_fingerprint=manifest.sap_fingerprint,
                            software_commit=attestation.software_commit,
                            state_fingerprint=state_fp,
                            parent_checkpoint_id=parent_checkpoint_id,
                            state_schema_version="1.0.0",
                            serialized_state=ser_state
                        )
                        self.checkpoints.save(ckpt)
                        parent_checkpoint_id = e.entry_id
                    else:
                        current_state = None
                        parent_checkpoint_id = None
                else:
                    continue

        return results
