"""researchforge/benchmarks/execution/protocol.py — Benchmark Execution Protocol & Dual-Domain Safety Gate.

Phase 12B.2 Core Invariants (Dual-Domain Cryptographic Hardening):
1. Complete Chain Verification (C <-> SAP <-> M <-> P <-> A):
   - Cohort canonical + physical file SHA-256
   - SAP canonical fingerprint
   - Manifest canonical + physical file SHA-256
   - Preflight report canonical + physical file SHA-256
   - Immutable Git commit SHA verification
   - Attestation self-signature anti-circular verification
2. Strict Separation Between Verification and Authorization:
   - Even if preflight_pass == True and all hashes match, execution is BLOCKED unless execution_authorized == True.
   - ExecutionNotAuthorizedError is raised if unauthorized execution is attempted.
3. Append-Only Execution Ledger:
   - Every completed task evaluation emits an auditable, hash-chained LedgerEntry.
4. Auditable Artifact Registry:
   - Tracks both canonical_fingerprint and physical_file_sha256 for all suite artifacts.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    ArtifactRegistryRecord,
    ExecutionManifest,
    ExecutionNotAuthorizedError,
    LedgerEntry,
    PreflightAttestation,
    PreflightGateError,
    PreflightValidationReport,
    RunManifestEntry,
    RunManifestEntry,
    get_software_commit_info,
)
from .preflight import PreflightValidator
from .engine import BenchmarkExecutionEngine


class BenchmarkExecutionProtocol:
    """Orchestrates preflight validation, authorization gating, task execution, and auditable artifact registry."""

    def __init__(
        self,
        validator: Optional[PreflightValidator] = None,
        ledger_path: Optional[Path | str] = None,
        registry_path: Optional[Path | str] = None,
    ) -> None:
        self.validator = validator or PreflightValidator()
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.registry_path = Path(registry_path) if registry_path else None
        self.evaluator = ContinuityEvaluator()

    # ── Preflight & Attestation Management ──────────────────────────────────────
    def run_preflight(
        self,
        manifest: ExecutionManifest,
        cohort_file_path: Optional[Path | str] = None,
        manifest_file_path: Optional[Path | str] = None,
        preflight_report_file_path: Optional[Path | str] = None,
        software_commit: Optional[str] = None,
        dirty_worktree: Optional[bool] = None,
        execution_authorized: bool = False,
        authorized_by: Optional[str] = None,
    ) -> Tuple[PreflightValidationReport, PreflightAttestation]:
        """Runs the 8-point preflight suite and generates a signed attestation token with dual binding."""
        report = self.validator.validate_all(manifest)

        cohort_sha = compute_file_sha256(cohort_file_path) if cohort_file_path and Path(cohort_file_path).exists() else ""
        manifest_sha = compute_file_sha256(manifest_file_path) if manifest_file_path and Path(manifest_file_path).exists() else ""
        report_sha = compute_file_sha256(preflight_report_file_path) if preflight_report_file_path and Path(preflight_report_file_path).exists() else ""

        attestation = self.validator.generate_attestation(
            report=report,
            cohort_file_sha256=cohort_sha,
            manifest_file_sha256=manifest_sha,
            preflight_report_file_sha256=report_sha,
            software_commit=software_commit,
            dirty_worktree=dirty_worktree,
            execution_authorized=execution_authorized,
            authorized_by=authorized_by,
            authorization_timestamp=datetime.now(timezone.utc).isoformat() if execution_authorized else None,
        )
        return report, attestation

    # ── Complete Chain Authorization Check ──────────────────────────────────────
    def verify_authorization(
        self,
        manifest: ExecutionManifest,
        attestation: PreflightAttestation,
        manifest_file_path: Optional[Path | str] = None,
        cohort_file_path: Optional[Path | str] = None,
        report_file_path: Optional[Path | str] = None,
        report: Optional[PreflightValidationReport] = None,
        verify_git_commit: bool = True,
    ) -> None:
        """Enforces the complete cryptographic authorization chain: C <-> SAP <-> M <-> P <-> A.
        
        Validates:
        1. Preflight pass == True
        2. Attestation self-signature anti-circular integrity
        3. Cohort canonical hash & physical file SHA-256
        4. SAP canonical hash
        5. Manifest canonical hash & physical file SHA-256
        6. Preflight report canonical hash & physical file SHA-256
        7. Immutable Git commit SHA binding
        8. Human authorization safety gate (execution_authorized == True)
        """
        # 1. Preflight verification status
        if not attestation.preflight_pass:
            raise PreflightGateError(
                "Benchmark execution blocked: Preflight verification gate has not passed."
            )

        # 2. Attestation self-hash integrity
        recomputed_attestation_fp = attestation.compute_fingerprint()
        if attestation.attestation_fingerprint and attestation.attestation_fingerprint != recomputed_attestation_fp:
            raise ExecutionNotAuthorizedError(
                f"Attestation cryptographic signature mismatch: token has been tampered with! "
                f"Recorded: {attestation.attestation_fingerprint}, Recomputed: {recomputed_attestation_fp}"
            )

        # 3. Cohort Binding (Canonical & Physical)
        cohort_spec = self.validator.cohort_spec
        if attestation.cohort_canonical_fingerprint != cohort_spec.cohort_fingerprint:
            raise ExecutionNotAuthorizedError(
                f"Attestation cohort canonical fingerprint ({attestation.cohort_canonical_fingerprint}) "
                f"does not match loaded cohort specification ({cohort_spec.cohort_fingerprint})."
            )

        if cohort_file_path and Path(cohort_file_path).exists():
            cohort_phys_sha = compute_file_sha256(cohort_file_path)
            if attestation.cohort_physical_file_sha256 and cohort_phys_sha != attestation.cohort_physical_file_sha256:
                raise ExecutionNotAuthorizedError(
                    f"Cohort physical file SHA-256 mismatch! On-disk: {cohort_phys_sha}, "
                    f"Attested: {attestation.cohort_physical_file_sha256}. File has been modified on disk."
                )

        # 4. SAP Binding (Canonical)
        sap_fp = getattr(self.validator.sap_spec, "plan_fingerprint", getattr(self.validator.sap_spec, "sap_fingerprint", ""))
        if attestation.sap_canonical_fingerprint != sap_fp:
            raise ExecutionNotAuthorizedError(
                f"Attestation SAP canonical fingerprint ({attestation.sap_canonical_fingerprint}) "
                f"does not match loaded SAP plan ({sap_fp})."
            )

        # 5. Manifest Binding (Canonical & Physical)
        recomputed_manifest_fp = manifest.compute_fingerprint()
        if manifest.manifest_fingerprint and manifest.manifest_fingerprint != recomputed_manifest_fp:
            raise ExecutionNotAuthorizedError(
                f"Manifest internal canonical fingerprint mismatch! Declared: {manifest.manifest_fingerprint}, "
                f"Recomputed from semantic entries: {recomputed_manifest_fp}. Manifest has been tampered with in memory!"
            )

        if attestation.manifest_canonical_fingerprint != manifest.manifest_fingerprint:
            raise ExecutionNotAuthorizedError(
                f"Attestation manifest canonical fingerprint ({attestation.manifest_canonical_fingerprint}) "
                f"does not match loaded manifest ({manifest.manifest_fingerprint})."
            )

        if manifest_file_path and Path(manifest_file_path).exists():
            manifest_phys_sha = compute_file_sha256(manifest_file_path)
            if attestation.manifest_physical_file_sha256 and manifest_phys_sha != attestation.manifest_physical_file_sha256:
                raise ExecutionNotAuthorizedError(
                    f"Manifest physical file SHA-256 mismatch! On-disk: {manifest_phys_sha}, "
                    f"Attested: {attestation.manifest_physical_file_sha256}. File has been modified on disk."
                )

        # 6. Preflight Report Binding (Canonical & Physical)
        if report and attestation.preflight_report_canonical_fingerprint != report.report_fingerprint:
            raise ExecutionNotAuthorizedError(
                f"Attestation preflight report canonical fingerprint ({attestation.preflight_report_canonical_fingerprint}) "
                f"does not match loaded report ({report.report_fingerprint})."
            )

        if report_file_path and Path(report_file_path).exists():
            report_phys_sha = compute_file_sha256(report_file_path)
            if attestation.preflight_report_physical_file_sha256 and report_phys_sha != attestation.preflight_report_physical_file_sha256:
                raise ExecutionNotAuthorizedError(
                    f"Preflight report physical file SHA-256 mismatch! On-disk: {report_phys_sha}, "
                    f"Attested: {attestation.preflight_report_physical_file_sha256}. File has been modified on disk."
                )

        # 7. Git Commit Verification
        if verify_git_commit and attestation.software_commit:
            current_git_sha, _ = get_software_commit_info()
            if attestation.software_commit != current_git_sha:
                raise ExecutionNotAuthorizedError(
                    f"Software commit SHA mismatch! Attestation was signed for commit {attestation.software_commit}, "
                    f"but runtime environment is at {current_git_sha}."
                )

        # 8. Human Authorization Safety Gate
        if not attestation.execution_authorized:
            raise ExecutionNotAuthorizedError(
                "Benchmark execution blocked: Preflight gates passed and complete cryptographic chain verified, "
                "but execution is NOT authorized. Human authorization token is required before committing 32,400-trial budget."
            )

    # ── Ledger Management ───────────────────────────────────────────────────────
    def append_ledger_entry(self, entry: LedgerEntry, ledger_path: Optional[Path | str] = None) -> None:
        """Appends a record to the append-only JSONL ledger."""
        path = Path(ledger_path) if ledger_path else self.ledger_path
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(_canonical_json(entry.to_dict()) + "\n")

    # ── Artifact Registry Management ───────────────────────────────────────────
    def register_artifact(
        self,
        artifact_name: str,
        file_path: Path | str,
        description: str,
        canonical_fingerprint: str = "",
        registry_path: Optional[Path | str] = None,
    ) -> ArtifactRegistryRecord:
        """Registers an artifact with dual-domain hashes into the registry file."""
        target = Path(file_path)
        if not target.exists():
            raise FileNotFoundError(f"Cannot register non-existent artifact: {target}")

        with open(target, "rb") as f:
            content = f.read()

        physical_sha256 = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)

        # Determine canonical fingerprint if not provided
        if not canonical_fingerprint:
            try:
                parsed = json.loads(content.decode("utf-8"))
                canonical_fingerprint = compute_canonical_fingerprint(parsed)
            except Exception:
                canonical_fingerprint = physical_sha256

        record = ArtifactRegistryRecord(
            artifact_name=artifact_name,
            file_path=str(target),
            canonical_fingerprint=canonical_fingerprint,
            physical_file_sha256=physical_sha256,
            size_bytes=size_bytes,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        reg_p = Path(registry_path) if registry_path else self.registry_path
        if reg_p:
            reg_p.parent.mkdir(parents=True, exist_ok=True)
            existing: Dict[str, Any] = {"artifacts": []}
            if reg_p.exists():
                try:
                    with open(reg_p, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    existing = {"artifacts": []}
            records = [r for r in existing.get("artifacts", []) if r.get("artifact_name") != artifact_name]
            records.append(record.to_dict())
            existing["artifacts"] = records
            with open(reg_p, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, sort_keys=True)

        return record

    # ── Single Task Evaluation ──────────────────────────────────────────────────
    # ── Benchmark Execution Runner (Safety Gated) ───────────────────────────────
    def execute_benchmark(
        self,
        manifest: ExecutionManifest,
        attestation: PreflightAttestation,
        manifest_file_path: Optional[Path | str] = None,
        cohort_file_path: Optional[Path | str] = None,
        sap_file_path: Optional[Path | str] = None,
        report_file_path: Optional[Path | str] = None,
        report: Optional[PreflightValidationReport] = None,
        checkpoints_dir: Optional[Path | str] = None,
        dry_run: bool = False
    ) -> List[LedgerEntry]:
        """Executes the full benchmark catalog.
        
        MANDATORY INVARIANT:
        This method delegates to BenchmarkExecutionEngine, which will immediately 
        raise ExecutionNotAuthorizedError unless the entire cryptographic chain matches 
        AND attestation.execution_authorized is explicitly True!
        """
        # Ensure we have paths
        ledger_p = self.ledger_path or Path("ledger.jsonl")
        registry_p = self.registry_path or Path("registry.json")
        check_p = Path(checkpoints_dir) if checkpoints_dir else ledger_p.parent / ".checkpoints"
        
        engine = BenchmarkExecutionEngine(
            ledger_path=ledger_p,
            registry_path=registry_p,
            checkpoints_dir=check_p
        )
        
        return engine.execute(
            manifest=manifest,
            attestation=attestation,
            manifest_file_path=manifest_file_path,
            cohort_file_path=cohort_file_path,
            sap_file_path=sap_file_path,
            dry_run=dry_run
        )
