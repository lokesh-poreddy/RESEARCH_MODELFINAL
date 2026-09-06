r"""researchforge/benchmarks/execution/models.py — Execution Protocol & Manifest Contracts.

Phase 12B.2 Core Invariants (Dual-Domain Cryptographic Hardening):
1. Dual-Domain Binding:
   - canonical_fingerprint: SHA-256 of canonical JSON semantic payload (excluding self-referential fingerprint).
   - physical_file_sha256: SHA-256 of exact bytes on filesystem.
2. Anti-Circularity Rule:
   - Every self-fingerprinted artifact computes H = SHA256(canonical_json(payload \ {self_fingerprint})).
3. Two-Tier Transfer Regimes:
   - sequence_transfer_regime: Structural relation between source tasks and target task.
   - condition_effective_regime: Actual experience exposed to the agent under condition semantics
     (e.g., COLD_START always has NO_PRIOR_EXPERIENCE).
4. Atomic Sequence Groups & Causal DAG:
   - Every entry belongs to a sequence_group_id (condition__ordering__seed).
   - Depends explicitly on strictly preceding entries in the same group.
5. Complete Chain Verification:
   - Attestation binds Cohort <-> SAP <-> Manifest <-> Preflight Report <-> Git Commit.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from ...domain.base import (
    _canonical_json,
    compute_canonical_fingerprint,
    compute_file_sha256,
)
from ..cohort.models import (
    BenchmarkCondition,
    RunOutcomeCategory,
    TaskFamily,
    TransferRegime,
)


PREREGISTERED_BENCHMARK_ARTIFACTS = frozenset([
    "phase12b_execution_manifest.json",
    "phase12b_preflight_report.json",
    "phase12b_preflight_attestation.json",
    "phase12b_artifact_registry.json",
    "phase12b_execution_ledger.jsonl",
    ".DS_Store",
])


def get_software_commit_info(repo_root: Optional[Path | str] = None) -> Tuple[str, bool]:
    """Retrieves the immutable 40-character Git commit SHA and worktree status.
    
    A worktree is considered clean (dirty=False) when git status --porcelain contains
    zero modified tracked files, zero staged modifications, and zero untracked files
    outside the explicitly preregistered benchmark execution artifacts.
    """
    root = Path(repo_root) if repo_root else Path(__file__).parent.parent.parent.parent
    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        git_sha = commit_res.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        raw_lines = [l.strip() for l in status_res.stdout.splitlines() if l.strip()]
        dirty_lines = []
        for line in raw_lines:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                path_str = parts[1].strip()
                if Path(path_str).name in PREREGISTERED_BENCHMARK_ARTIFACTS:
                    continue
            dirty_lines.append(line)

        is_dirty = len(dirty_lines) > 0
        return git_sha, is_dirty
    except Exception:
        return "0000000000000000000000000000000000000000", True


class PreflightGateError(Exception):
    """Raised when one or more preflight verification gates fail."""
    pass


class ExecutionNotAuthorizedError(Exception):
    """Raised when an attempt is made to execute benchmark runs without an authorized attestation."""
    pass


class CheckpointInvalidError(Exception):
    """Raised when an ExecutionCheckpoint fails validation against sealed artifacts or state fingerprint."""
    pass


class ManifestInvalidError(Exception):
    """Raised when the ExecutionManifest contains DAG cycles, duplicate entries, or invalid dependencies."""
    pass


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class DependencyPolicy(str, Enum):
    EXECUTION_COMPLETE = "EXECUTION_COMPLETE"
    SCIENTIFICALLY_VALID = "SCIENTIFICALLY_VALID"
    ARTIFACT_AVAILABLE = "ARTIFACT_AVAILABLE"


@dataclass(frozen=True)
class RunManifestEntry:
    """Immutable execution contract for a single task evaluation inside a sequence group."""
    entry_id: str
    condition: BenchmarkCondition
    seed: int
    ordering: str
    sequence_group_id: str
    sequence_position: int
    task_id: str
    task_family: TaskFamily
    task_fingerprint: str
    prior_task_ids: Tuple[str, ...]
    prior_task_families: Tuple[TaskFamily, ...]
    sequence_transfer_regime: TransferRegime
    condition_effective_regime: TransferRegime
    depends_on_entry_ids: Tuple[str, ...]
    dependency_policy: DependencyPolicy = DependencyPolicy.EXECUTION_COMPLETE
    generation_budget: int = 10
    population_size: int = 6
    timeout_seconds: float = 30.0
    cohort_fingerprint: str = ""
    sap_fingerprint: str = ""
    entry_fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        """Computes canonical fingerprint excluding self-referential entry_fingerprint."""
        return compute_canonical_fingerprint(self.to_dict(), ["entry_fingerprint"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "condition": self.condition.value,
            "seed": self.seed,
            "ordering": self.ordering,
            "sequence_group_id": self.sequence_group_id,
            "sequence_position": self.sequence_position,
            "task_id": self.task_id,
            "task_family": self.task_family.value,
            "task_fingerprint": self.task_fingerprint,
            "prior_task_ids": list(self.prior_task_ids),
            "prior_task_families": [f.value for f in self.prior_task_families],
            "sequence_transfer_regime": self.sequence_transfer_regime.value,
            "condition_effective_regime": self.condition_effective_regime.value,
            "depends_on_entry_ids": list(self.depends_on_entry_ids),
            "dependency_policy": self.dependency_policy.value,
            "generation_budget": self.generation_budget,
            "population_size": self.population_size,
            "timeout_seconds": self.timeout_seconds,
            "cohort_fingerprint": self.cohort_fingerprint,
            "sap_fingerprint": self.sap_fingerprint,
            "entry_fingerprint": self.entry_fingerprint,
        }


@dataclass(frozen=True)
class ExecutionManifest:
    """Immutable, content-addressed catalog of all planned task evaluations across the benchmark."""
    manifest_version: str = "1.0.0"
    cohort_version: str = "1.0.1"
    cohort_fingerprint: str = ""
    sap_version: str = "1.0.0"
    sap_fingerprint: str = ""
    researchforge_release: str = "1.0.0-alpha.3"
    git_commit: str = "d4ddad67ea5a22aea15cdbb69c984e3a7e120483"
    environment_fingerprint: str = ""
    entry_count: int = 540
    planned_trials: int = 32400
    sequence_group_count: int = 90
    entries: Tuple[RunManifestEntry, ...] = field(default_factory=tuple)
    manifest_fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        """Computes canonical fingerprint excluding self-referential manifest_fingerprint."""
        payload = {
            "manifest_version": self.manifest_version,
            "cohort_version": self.cohort_version,
            "cohort_fingerprint": self.cohort_fingerprint,
            "sap_version": self.sap_version,
            "sap_fingerprint": self.sap_fingerprint,
            "researchforge_release": self.researchforge_release,
            "git_commit": self.git_commit,
            "environment_fingerprint": self.environment_fingerprint,
            "entry_count": self.entry_count,
            "planned_trials": self.planned_trials,
            "sequence_group_count": self.sequence_group_count,
            "entry_fingerprints": [e.entry_fingerprint or e.compute_fingerprint() for e in self.entries],
        }
        return compute_canonical_fingerprint(payload, ["manifest_fingerprint"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "cohort_version": self.cohort_version,
            "cohort_fingerprint": self.cohort_fingerprint,
            "sap_version": self.sap_version,
            "sap_fingerprint": self.sap_fingerprint,
            "researchforge_release": self.researchforge_release,
            "git_commit": self.git_commit,
            "environment_fingerprint": self.environment_fingerprint,
            "entry_count": self.entry_count,
            "planned_trials": self.planned_trials,
            "sequence_group_count": self.sequence_group_count,
            "entries": [e.to_dict() for e in self.entries],
            "manifest_fingerprint": self.manifest_fingerprint or self.compute_fingerprint(),
        }


@dataclass(frozen=True)
class PreflightCheckResult:
    """Outcome of an individual preflight gate check."""
    gate_number: int
    gate_name: str
    status: str  # "PASSED" or "FAILED"
    details: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_number": self.gate_number,
            "gate_name": self.gate_name,
            "status": self.status,
            "details": self.details,
            "evidence": self.evidence,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class PreflightValidationReport:
    """Aggregated verification report covering all 8 preflight gates."""
    report_id: str
    report_version: str = "1.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cohort_fingerprint: str = ""
    sap_fingerprint: str = ""
    manifest_fingerprint: str = ""
    all_gates_passed: bool = False
    gates: Tuple[PreflightCheckResult, ...] = field(default_factory=tuple)
    report_fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        """Computes canonical fingerprint excluding self-referential report_fingerprint."""
        payload = {
            "report_id": self.report_id,
            "report_version": self.report_version,
            "timestamp": self.timestamp,
            "cohort_fingerprint": self.cohort_fingerprint,
            "sap_fingerprint": self.sap_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "all_gates_passed": self.all_gates_passed,
            "gates": [g.to_dict() for g in self.gates],
        }
        return compute_canonical_fingerprint(payload, ["report_fingerprint"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_version": self.report_version,
            "timestamp": self.timestamp,
            "cohort_fingerprint": self.cohort_fingerprint,
            "sap_fingerprint": self.sap_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "all_gates_passed": self.all_gates_passed,
            "gates": [g.to_dict() for g in self.gates],
            "report_fingerprint": self.report_fingerprint or self.compute_fingerprint(),
        }


@dataclass(frozen=True)
class PreflightAttestation:
    r"""Cryptographic attestation token binding the complete verification chain to authorization.
    
    CRITICAL INVARIANTS:
    1. Dual Binding: Holds both canonical object fingerprints (semantic payload) and physical
       file SHA-256 hashes (exact disk bytes) for Cohort, Manifest, and Preflight Report.
    2. Anti-Circularity: attestation_fingerprint is strictly computed over the unsigned payload:
       attestation_fingerprint = SHA256(canonical_json(payload \ {attestation_fingerprint})).
    3. Immutable Software Identity: Binds exact 40-character Git commit SHA, not mutable 'HEAD'.
    4. Safety Gate: preflight_pass != execution_authorized.
    """
    attestation_version: str = "1.1.0"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Complete dual binding chain
    cohort_canonical_fingerprint: str = ""
    cohort_physical_file_sha256: str = ""

    sap_canonical_fingerprint: str = ""

    manifest_canonical_fingerprint: str = ""
    manifest_physical_file_sha256: str = ""

    preflight_report_canonical_fingerprint: str = ""
    preflight_report_physical_file_sha256: str = ""

    # Software commit identity
    software_commit: str = ""
    dirty_worktree: bool = False

    # Authorization status
    preflight_pass: bool = False
    execution_authorized: bool = False
    authorized_by: Optional[str] = None
    authorization_timestamp: Optional[str] = None
    attestation_fingerprint: str = ""

    # Backward compatibility properties
    @property
    def cohort_fingerprint(self) -> str:
        return self.cohort_canonical_fingerprint

    @property
    def sap_fingerprint(self) -> str:
        return self.sap_canonical_fingerprint

    @property
    def manifest_fingerprint(self) -> str:
        return self.manifest_canonical_fingerprint

    @property
    def preflight_report_fingerprint(self) -> str:
        return self.preflight_report_canonical_fingerprint

    def __post_init__(self):
        if isinstance(self.software_commit, dict):
            git_sha = self.software_commit.get("git_sha", "")
            dirty = self.software_commit.get("dirty_worktree", self.dirty_worktree)
            object.__setattr__(self, "software_commit", str(git_sha))
            object.__setattr__(self, "dirty_worktree", bool(dirty))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreflightAttestation:
        commit_val = data.get("software_commit")
        if isinstance(commit_val, dict):
            commit_sha = str(commit_val.get("git_sha", ""))
            dirty_worktree = bool(commit_val.get("dirty_worktree", data.get("dirty_worktree", False)))
        else:
            commit_sha = str(commit_val or data.get("software_commit_sha", ""))
            dirty_worktree = bool(data.get("dirty_worktree", False))

        cohort_data = data.get("cohort", {})
        cohort_canon = cohort_data.get("canonical_fingerprint", data.get("cohort_canonical_fingerprint", ""))
        cohort_phys = cohort_data.get("physical_file_sha256", data.get("cohort_physical_file_sha256", ""))

        sap_data = data.get("sap", {})
        sap_canon = sap_data.get("canonical_fingerprint", data.get("sap_canonical_fingerprint", ""))

        manifest_data = data.get("manifest", {})
        manifest_canon = manifest_data.get("canonical_fingerprint", data.get("manifest_canonical_fingerprint", ""))
        manifest_phys = manifest_data.get("physical_file_sha256", data.get("manifest_physical_file_sha256", ""))

        report_data = data.get("preflight_report", {})
        report_canon = report_data.get("canonical_fingerprint", data.get("preflight_report_canonical_fingerprint", ""))
        report_phys = report_data.get("physical_file_sha256", data.get("preflight_report_physical_file_sha256", ""))

        return cls(
            attestation_version=data.get("attestation_version", "1.1.0"),
            timestamp=data.get("timestamp", ""),
            cohort_canonical_fingerprint=cohort_canon,
            cohort_physical_file_sha256=cohort_phys,
            sap_canonical_fingerprint=sap_canon,
            manifest_canonical_fingerprint=manifest_canon,
            manifest_physical_file_sha256=manifest_phys,
            preflight_report_canonical_fingerprint=report_canon,
            preflight_report_physical_file_sha256=report_phys,
            software_commit=commit_sha,
            dirty_worktree=dirty_worktree,
            preflight_pass=data.get("preflight_pass", False),
            execution_authorized=data.get("execution_authorized", False),
            authorized_by=data.get("authorized_by"),
            authorization_timestamp=data.get("authorization_timestamp"),
            attestation_fingerprint=data.get("attestation_fingerprint", ""),
        )

    def to_unsigned_dict(self) -> Dict[str, Any]:
        """Returns the unsigned payload dictionary strictly excluding attestation_fingerprint."""
        commit_sha = self.software_commit.get("git_sha", "") if isinstance(self.software_commit, dict) else self.software_commit
        dirty = self.software_commit.get("dirty_worktree", self.dirty_worktree) if isinstance(self.software_commit, dict) else self.dirty_worktree
        return {
            "attestation_version": self.attestation_version,
            "timestamp": self.timestamp,
            "cohort": {
                "canonical_fingerprint": self.cohort_canonical_fingerprint,
                "physical_file_sha256": self.cohort_physical_file_sha256,
            },
            "sap": {
                "canonical_fingerprint": self.sap_canonical_fingerprint,
            },
            "manifest": {
                "canonical_fingerprint": self.manifest_canonical_fingerprint,
                "physical_file_sha256": self.manifest_physical_file_sha256,
            },
            "preflight_report": {
                "canonical_fingerprint": self.preflight_report_canonical_fingerprint,
                "physical_file_sha256": self.preflight_report_physical_file_sha256,
            },
            "software_commit": {
                "git_sha": commit_sha,
                "dirty_worktree": dirty,
            },
            "cohort_canonical_fingerprint": self.cohort_canonical_fingerprint,
            "cohort_physical_file_sha256": self.cohort_physical_file_sha256,
            "sap_canonical_fingerprint": self.sap_canonical_fingerprint,
            "manifest_canonical_fingerprint": self.manifest_canonical_fingerprint,
            "manifest_physical_file_sha256": self.manifest_physical_file_sha256,
            "preflight_report_canonical_fingerprint": self.preflight_report_canonical_fingerprint,
            "preflight_report_physical_file_sha256": self.preflight_report_physical_file_sha256,
            "software_commit_sha": commit_sha,
            "dirty_worktree": dirty,
            "preflight_pass": self.preflight_pass,
            "execution_authorized": self.execution_authorized,
            "authorized_by": self.authorized_by,
            "authorization_timestamp": self.authorization_timestamp,
        }

    def compute_fingerprint(self) -> str:
        """Computes canonical fingerprint of attestation strictly over the unsigned payload."""
        return compute_canonical_fingerprint(self.to_unsigned_dict(), ["attestation_fingerprint"])

    def to_dict(self) -> Dict[str, Any]:
        d = self.to_unsigned_dict()
        d["attestation_fingerprint"] = self.attestation_fingerprint or self.compute_fingerprint()
        return d


@dataclass(frozen=True)
class ArtifactRegistryRecord:
    """Auditable record for an artifact in the Phase 12B suite with dual-domain hashes."""
    artifact_name: str
    file_path: str
    canonical_fingerprint: str
    physical_file_sha256: str
    size_bytes: int
    description: str
    created_at: str

    @property
    def sha256_hash(self) -> str:
        """Backward compatibility alias for physical_file_sha256."""
        return self.physical_file_sha256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_name": self.artifact_name,
            "file_path": self.file_path,
            "canonical_fingerprint": self.canonical_fingerprint,
            "physical_file_sha256": self.physical_file_sha256,
            "sha256_hash": self.physical_file_sha256,
            "size_bytes": self.size_bytes,
            "description": self.description,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class LedgerEntry:
    """Append-only record of an individual completed task run."""
    timestamp: str
    entry_id: str
    sequence_group_id: str
    sequence_position: int
    condition: str
    task_id: str
    seed: int
    ordering: str
    execution_status: str
    outcome_status: str
    best_metric: float
    decision_quality: float
    trials_executed: int
    wallclock_seconds: float
    memory_fingerprint_before: str
    memory_fingerprint_after: str
    entry_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "entry_id": self.entry_id,
            "sequence_group_id": self.sequence_group_id,
            "sequence_position": self.sequence_position,
            "condition": self.condition,
            "task_id": self.task_id,
            "seed": self.seed,
            "ordering": self.ordering,
            "execution_status": self.execution_status,
            "outcome_status": self.outcome_status,
            "best_metric": self.best_metric,
            "decision_quality": self.decision_quality,
            "trials_executed": self.trials_executed,
            "wallclock_seconds": self.wallclock_seconds,
            "memory_fingerprint_before": self.memory_fingerprint_before,
            "memory_fingerprint_after": self.memory_fingerprint_after,
            "entry_hash": self.entry_hash,
        }


@dataclass(frozen=True)
class ExecutionCheckpoint:
    """Canonical, self-validating state checkpoint for resumption without opaque pickling."""
    entry_id: str
    sequence_group_id: str
    sequence_position: int
    manifest_fingerprint: str
    cohort_fingerprint: str
    sap_fingerprint: str
    software_commit: str
    state_fingerprint: str
    parent_checkpoint_id: Optional[str]
    state_schema_version: str
    serialized_state: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate_integrity(
        self,
        sealed_manifest_fp: str,
        sealed_cohort_fp: str,
        sealed_sap_fp: str,
        execution_commit: str
    ) -> None:
        """Validates that this checkpoint belongs to the exact sealed artifacts."""
        if self.manifest_fingerprint != sealed_manifest_fp:
            raise CheckpointInvalidError("Checkpoint manifest fingerprint mismatch.")
        if self.cohort_fingerprint != sealed_cohort_fp:
            raise CheckpointInvalidError("Checkpoint cohort fingerprint mismatch.")
        if self.sap_fingerprint != sealed_sap_fp:
            raise CheckpointInvalidError("Checkpoint SAP fingerprint mismatch.")
        if self.software_commit != execution_commit:
            raise CheckpointInvalidError("Checkpoint software commit mismatch.")
        
        # Verify state fingerprint
        recomputed = compute_canonical_fingerprint(self.serialized_state)
        if self.state_fingerprint != recomputed:
            raise CheckpointInvalidError(f"Checkpoint state fingerprint mismatch: {self.state_fingerprint} != {recomputed}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "sequence_group_id": self.sequence_group_id,
            "sequence_position": self.sequence_position,
            "manifest_fingerprint": self.manifest_fingerprint,
            "cohort_fingerprint": self.cohort_fingerprint,
            "sap_fingerprint": self.sap_fingerprint,
            "software_commit": self.software_commit,
            "state_fingerprint": self.state_fingerprint,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "state_schema_version": self.state_schema_version,
            "serialized_state": self.serialized_state,
            "created_at": self.created_at,
        }
