"""Phase 13.10 RF-FLEX execution gate and pre-execution attestation.

This module prepares and verifies the execution identity. It deliberately does
not launch the confirmatory experiment unless a human-authorized attestation
binds the frozen protocol, task manifest, exact commit, and clean worktree.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .continuity.tasks import get_full_task_sequence
from .rf_flex_confirmatory import RFFlexConfirmatoryProtocol


class RFExecutionGateError(RuntimeError):
    """Raised when a confirmatory execution gate fails."""


class RFExecutionNotAuthorizedError(RFExecutionGateError):
    """Raised when execution is attempted without explicit authorization."""


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip())
        if len(commit) != 40:
            raise RFExecutionGateError("Git commit identity is not a full SHA-1")
        return commit, dirty
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RFExecutionGateError("Unable to determine exact git execution identity") from exc


def task_manifest(repo_root: Path | str) -> Dict[str, Any]:
    tasks = get_full_task_sequence(seed=0)
    rows = [task.to_dict() | {"task_fingerprint": task.fingerprint()} for task in tasks]
    payload = {
        "manifest_version": "13.10.0",
        "tasks": rows,
        "task_ids": [task.task_id for task in tasks],
    }
    return {**payload, "manifest_fingerprint": _fingerprint(payload)}


def environment_identity(repo_root: Path | str) -> Dict[str, Any]:
    root = Path(repo_root)
    requirements = root / "requirements.txt"
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "requirements_sha256": _file_sha256(requirements) if requirements.exists() else None,
    }
    return {**payload, "environment_fingerprint": _fingerprint(payload)}


@dataclass(frozen=True)
class RFPreExecutionAttestation:
    attestation_version: str
    created_at: str
    phase13_9_protocol_fingerprint: str
    protocol_file_sha256: str
    task_manifest_fingerprint: str
    software_commit: str
    dirty_worktree: bool
    environment: Dict[str, Any]
    planned_trajectory_groups: int
    planned_conditions: int
    planned_tasks: int
    planned_seeds: int
    planned_orderings: int
    statistical_analysis_during_execution: bool
    authorization: bool
    authorized_by: Optional[str]
    authorization_timestamp: Optional[str]
    preflight_pass: bool
    attestation_fingerprint: str = ""

    def unsigned_dict(self) -> Dict[str, Any]:
        return {
            "attestation_version": self.attestation_version,
            "created_at": self.created_at,
            "phase13_9_protocol_fingerprint": self.phase13_9_protocol_fingerprint,
            "protocol_file_sha256": self.protocol_file_sha256,
            "task_manifest_fingerprint": self.task_manifest_fingerprint,
            "software_commit": self.software_commit,
            "dirty_worktree": self.dirty_worktree,
            "environment": self.environment,
            "planned_trajectory_groups": self.planned_trajectory_groups,
            "planned_conditions": self.planned_conditions,
            "planned_tasks": self.planned_tasks,
            "planned_seeds": self.planned_seeds,
            "planned_orderings": self.planned_orderings,
            "statistical_analysis_during_execution": self.statistical_analysis_during_execution,
            "authorization": self.authorization,
            "authorized_by": self.authorized_by,
            "authorization_timestamp": self.authorization_timestamp,
            "preflight_pass": self.preflight_pass,
        }

    def fingerprint(self) -> str:
        return _fingerprint(self.unsigned_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attestation_version": self.attestation_version,
            "created_at": self.created_at,
            "phase13_9_protocol_fingerprint": self.phase13_9_protocol_fingerprint,
            "protocol_file_sha256": self.protocol_file_sha256,
            "task_manifest_fingerprint": self.task_manifest_fingerprint,
            "software_commit": self.software_commit,
            "dirty_worktree": self.dirty_worktree,
            "environment": self.environment,
            "planned_trajectory_groups": self.planned_trajectory_groups,
            "planned_conditions": self.planned_conditions,
            "planned_tasks": self.planned_tasks,
            "planned_seeds": self.planned_seeds,
            "planned_orderings": self.planned_orderings,
            "statistical_analysis_during_execution": self.statistical_analysis_during_execution,
            "authorization": self.authorization,
            "authorized_by": self.authorized_by,
            "authorization_timestamp": self.authorization_timestamp,
            "preflight_pass": self.preflight_pass,
            "attestation_fingerprint": self.attestation_fingerprint or self.fingerprint(),
        }

    def write(self, output_path: Path | str) -> Dict[str, Any]:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RFPreExecutionAttestation":
        return cls(**data)


@dataclass(frozen=True)
class RFExecutionReleaseSeal:
    """Clean-commit release seal; authorization remains permanently false here."""

    seal_version: str
    created_at: str
    protocol_fingerprint: str
    software_commit: str
    dirty_worktree: bool
    task_manifest_fingerprint: str
    environment_fingerprint: str
    artifact_hashes: Dict[str, str]
    planned_trajectory_groups: int
    planned_conditions: int
    planned_tasks: int
    planned_seeds: int
    planned_orderings: int
    pre_execution_attestation_fingerprint: str
    execution_authorized: bool = False
    seal_fingerprint: str = ""

    def unsigned_dict(self) -> Dict[str, Any]:
        return {
            "seal_version": self.seal_version,
            "created_at": self.created_at,
            "protocol_fingerprint": self.protocol_fingerprint,
            "software_commit": self.software_commit,
            "dirty_worktree": self.dirty_worktree,
            "task_manifest_fingerprint": self.task_manifest_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "artifact_hashes": dict(self.artifact_hashes),
            "planned_trajectory_groups": self.planned_trajectory_groups,
            "planned_conditions": self.planned_conditions,
            "planned_tasks": self.planned_tasks,
            "planned_seeds": self.planned_seeds,
            "planned_orderings": self.planned_orderings,
            "pre_execution_attestation_fingerprint": self.pre_execution_attestation_fingerprint,
            "execution_authorized": self.execution_authorized,
        }

    def fingerprint(self) -> str:
        return _fingerprint(self.unsigned_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seal_version": self.seal_version,
            "created_at": self.created_at,
            "protocol_fingerprint": self.protocol_fingerprint,
            "software_commit": self.software_commit,
            "dirty_worktree": self.dirty_worktree,
            "task_manifest_fingerprint": self.task_manifest_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "artifact_hashes": dict(self.artifact_hashes),
            "planned_trajectory_groups": self.planned_trajectory_groups,
            "planned_conditions": self.planned_conditions,
            "planned_tasks": self.planned_tasks,
            "planned_seeds": self.planned_seeds,
            "planned_orderings": self.planned_orderings,
            "pre_execution_attestation_fingerprint": self.pre_execution_attestation_fingerprint,
            "execution_authorized": self.execution_authorized,
            "seal_fingerprint": self.seal_fingerprint or self.fingerprint(),
        }

    def write(self, output_path: Path | str) -> Dict[str, Any]:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return data


def prepare_attestation(protocol_path: Path | str, repo_root: Path | str) -> RFPreExecutionAttestation:
    root = Path(repo_root)
    protocol_file = Path(protocol_path)
    protocol_data = json.loads(protocol_file.read_text(encoding="utf-8"))
    protocol = RFFlexConfirmatoryProtocol(
        protocol_version=protocol_data["protocol_version"],
        status="DESIGN_ONLY",
    )
    protocol.validate()
    if protocol_data.get("protocol_fingerprint") != protocol.fingerprint():
        raise RFExecutionGateError("Frozen Phase 13.9 protocol fingerprint mismatch")
    commit, dirty = _git_state(root)
    manifest = task_manifest(root)
    environment = environment_identity(root)
    planned_groups = len(protocol.conditions) * len(protocol.orderings) * len(protocol.seeds) * len(protocol.task_universe)
    unsigned = RFPreExecutionAttestation(
        attestation_version="13.10.0",
        created_at=datetime.now(timezone.utc).isoformat(),
        phase13_9_protocol_fingerprint=protocol.fingerprint(),
        protocol_file_sha256=_file_sha256(protocol_file),
        task_manifest_fingerprint=manifest["manifest_fingerprint"],
        software_commit=commit,
        dirty_worktree=dirty,
        environment=environment,
        planned_trajectory_groups=planned_groups,
        planned_conditions=len(protocol.conditions),
        planned_tasks=len(protocol.task_universe),
        planned_seeds=len(protocol.seeds),
        planned_orderings=len(protocol.orderings),
        statistical_analysis_during_execution=False,
        authorization=False,
        authorized_by=None,
        authorization_timestamp=None,
        preflight_pass=not dirty,
    )
    return replace(unsigned, attestation_fingerprint=unsigned.fingerprint())


def authorize_attestation(attestation: RFPreExecutionAttestation, authorized_by: str) -> RFPreExecutionAttestation:
    if not authorized_by.strip():
        raise RFExecutionGateError("Human authorization identity is required")
    if not attestation.preflight_pass or attestation.dirty_worktree:
        raise RFExecutionGateError("Cannot authorize an attestation with a failed clean-worktree preflight")
    authorized = replace(
        attestation,
        authorization=True,
        authorized_by=authorized_by,
        authorization_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return replace(authorized, attestation_fingerprint=authorized.fingerprint())


def create_release_seal(
    protocol_path: Path | str,
    repo_root: Path | str,
    output_dir: Path | str,
) -> RFExecutionReleaseSeal:
    """Regenerate a clean-commit attestation and create an unauthorized release seal."""
    root = Path(repo_root)
    output = Path(output_dir)
    protocol_file = Path(protocol_path)
    attestation = prepare_attestation(protocol_file, root)
    if not attestation.preflight_pass or attestation.dirty_worktree:
        raise RFExecutionGateError(
            "RF-FLEX release seal requires a clean worktree; no seal was created"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = task_manifest(root)
    manifest_file = output / "rf_flex_13_10_task_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    attestation_file = output / "rf_flex_13_10_pre_execution_attestation.json"
    attestation.write(attestation_file)
    environment = environment_identity(root)
    seal = RFExecutionReleaseSeal(
        seal_version="13.11.0",
        created_at=datetime.now(timezone.utc).isoformat(),
        protocol_fingerprint=attestation.phase13_9_protocol_fingerprint,
        software_commit=attestation.software_commit,
        dirty_worktree=False,
        task_manifest_fingerprint=manifest["manifest_fingerprint"],
        environment_fingerprint=environment["environment_fingerprint"],
        artifact_hashes={
            "protocol_file_sha256": attestation.protocol_file_sha256,
            "task_manifest_file_sha256": _file_sha256(manifest_file),
            "attestation_file_sha256": _file_sha256(attestation_file),
        },
        planned_trajectory_groups=attestation.planned_trajectory_groups,
        planned_conditions=attestation.planned_conditions,
        planned_tasks=attestation.planned_tasks,
        planned_seeds=attestation.planned_seeds,
        planned_orderings=attestation.planned_orderings,
        pre_execution_attestation_fingerprint=attestation.attestation_fingerprint,
        execution_authorized=False,
    )
    return replace(seal, seal_fingerprint=seal.fingerprint())


def assert_execution_ready(attestation: RFPreExecutionAttestation, protocol: RFFlexConfirmatoryProtocol) -> None:
    protocol.validate()
    if not attestation.preflight_pass or attestation.dirty_worktree:
        raise RFExecutionNotAuthorizedError("RF-FLEX execution blocked: preflight or clean-worktree gate failed")
    if not attestation.authorization:
        raise RFExecutionNotAuthorizedError("RF-FLEX execution blocked: explicit human authorization is absent")
    if attestation.phase13_9_protocol_fingerprint != protocol.fingerprint():
        raise RFExecutionGateError("RF-FLEX execution blocked: protocol fingerprint mismatch")
    if attestation.statistical_analysis_during_execution:
        raise RFExecutionGateError("RF-FLEX execution blocked: statistical analysis during execution is enabled")
    if attestation.attestation_fingerprint != attestation.fingerprint():
        raise RFExecutionGateError("RF-FLEX execution blocked: attestation fingerprint mismatch")


def execute_authorized(
    attestation: RFPreExecutionAttestation,
    protocol: RFFlexConfirmatoryProtocol,
    runner: Callable[[], Any],
) -> Any:
    """Run a caller-provided immutable-record runner only after all gates pass."""
    assert_execution_ready(attestation, protocol)
    return runner()
