"""scripts/generate_phase12b_preflight_artifacts.py — Generates Phase 12B.2 Sealed Execution Artifacts.

Acyclic Dependency Hierarchy:
Cohort Specification ───┐
SAP Specification ──────┤
                        ↓
                 Execution Manifest
                        ↓
                 Preflight Report
                        ↓
                 Preflight Attestation (Dual-Domain Bound)
                        ↓
                 Artifact Registry (Downstream Index)

CRITICAL INVARIANTS:
1. Dual-domain binding: canonical object fingerprints and physical file SHA-256 hashes.
2. Anti-circularity: attestation_fingerprint excludes itself from hashed payload.
3. Software identity: binds exact 40-character Git commit SHA.
4. STOP GATE ACTIVE: DOES NOT execute the 32,400-trial benchmark!
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from researchforge.domain.base import compute_file_sha256
from researchforge.benchmarks.cohort import create_canonical_cohort_spec
from researchforge.benchmarks.execution import (
    BenchmarkExecutionProtocol,
    PreflightValidator,
    generate_execution_manifest,
    get_software_commit_info,
)
from researchforge.benchmarks.statistical import create_canonical_sap


def main():
    root_dir = Path(__file__).parent.parent
    cohort_spec = create_canonical_cohort_spec()
    sap = create_canonical_sap()
    cohort_path = root_dir / "phase12b_cohort_specification.json"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort specification file not found at {cohort_path}")
    cohort_phys_sha = compute_file_sha256(cohort_path)

    git_sha, is_dirty = get_software_commit_info(root_dir)

    print("================================================================================")
    print("           RESEARCHFORGE-ECRM: PHASE 12B.2 PREFLIGHT GATE EXECUTION             ")
    print("================================================================================")
    print(f"Software Commit:    {git_sha} (dirty={is_dirty})")
    print(f"Cohort Version:     {cohort_spec.specification_version}")
    print(f"Cohort Canonical:   {cohort_spec.cohort_fingerprint}")
    print(f"Cohort Physical:    {cohort_phys_sha}")
    print(f"SAP Version:        {sap.plan_version}")
    print(f"SAP Canonical:      {sap.plan_fingerprint}")
    print("--------------------------------------------------------------------------------")

    # [1/4] Generate Execution Manifest
    print("[1/4] Generating 540-entry Execution Manifest...")
    manifest = generate_execution_manifest(cohort_spec, sap)
    manifest_path = root_dir / "phase12b_execution_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    manifest_phys_sha = compute_file_sha256(manifest_path)
    print(f"  -> Canonical Fingerprint: {manifest.manifest_fingerprint}")
    print(f"  -> Physical File SHA-256: {manifest_phys_sha}")
    print(f"  -> Saved to: {manifest_path.name} ({manifest_path.stat().st_size} bytes)")

    # [2/4] Run Preflight Validator (8 Gates)
    print("\n[2/4] Executing 8-Point Preflight Verification Gate...")
    validator = PreflightValidator(cohort_spec, sap)
    report = validator.validate_all(manifest)
    report_path = root_dir / "phase12b_preflight_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    report_phys_sha = compute_file_sha256(report_path)

    for g in report.gates:
        status_symbol = "✓" if g.status == "PASSED" else "✗"
        print(f"  [{status_symbol}] Gate {g.gate_number}: {g.gate_name:<40} [{g.status}] ({g.latency_ms:.1f}ms)")

    print(f"  -> All Gates Passed:      {report.all_gates_passed}")
    print(f"  -> Canonical Fingerprint: {report.report_fingerprint}")
    print(f"  -> Physical File SHA-256: {report_phys_sha}")
    print(f"  -> Saved to: {report_path.name}")

    if not report.all_gates_passed:
        print("\nFATAL: One or more preflight gates failed. Aborting.")
        return 1

    # [3/4] Generate Dual-Bound Cryptographic Attestation Token
    print("\n[3/4] Generating Preflight Attestation Token (Dual-Bound, Human Authorization Pending)...")
    attestation = validator.generate_attestation(
        report=report,
        cohort_file_sha256=cohort_phys_sha,
        manifest_file_sha256=manifest_phys_sha,
        preflight_report_file_sha256=report_phys_sha,
        software_commit=git_sha,
        dirty_worktree=is_dirty,
        execution_authorized=False,
        authorized_by=None,
    )
    attestation_path = root_dir / "phase12b_preflight_attestation.json"
    with open(attestation_path, "w", encoding="utf-8") as f:
        json.dump(attestation.to_dict(), f, indent=2)
    attestation_phys_sha = compute_file_sha256(attestation_path)
    print(f"  -> Preflight Pass:        {attestation.preflight_pass}")
    print(f"  -> Execution Authorized:  {attestation.execution_authorized} (MANDATORY SAFETY LOCK)")
    print(f"  -> Attestation Token FP:  {attestation.attestation_fingerprint}")
    print(f"  -> Physical File SHA-256: {attestation_phys_sha}")
    print(f"  -> Saved to: {attestation_path.name}")

    # [4/4] Build Downstream Artifact Registry
    print("\n[4/4] Updating Auditable Artifact Registry...")
    registry_path = root_dir / "phase12b_artifact_registry.json"
    proto = BenchmarkExecutionProtocol(validator=validator, registry_path=registry_path)

    # Register suite artifacts with dual-domain hashes
    proto.register_artifact(
        artifact_name="cohort_specification",
        file_path=cohort_path,
        description="Frozen Phase 12B Benchmark Cohort Specification v1.0.1",
        canonical_fingerprint=cohort_spec.cohort_fingerprint,
    )
    proto.register_artifact(
        artifact_name="execution_manifest",
        file_path=manifest_path,
        description="Complete 540-entry task execution catalog with two-tier transfer regimes (32,400 planned trials)",
        canonical_fingerprint=manifest.manifest_fingerprint,
    )
    proto.register_artifact(
        artifact_name="preflight_validation_report",
        file_path=report_path,
        description="Exhaustive 8-point preflight verification report and gate evidence",
        canonical_fingerprint=report.report_fingerprint,
    )
    proto.register_artifact(
        artifact_name="preflight_attestation",
        file_path=attestation_path,
        description="Cryptographic preflight attestation token binding verification to authorization safety gate",
        canonical_fingerprint=attestation.attestation_fingerprint,
    )
    print(f"  -> Artifact Registry updated at: {registry_path.name}")

    print("\n================================================================================")
    print("                     PHASE 12B.2 PREFLIGHT STATUS: SEALED                       ")
    print("--------------------------------------------------------------------------------")
    print("STOP GATE ACTIVE: 32,400-trial benchmark execution is deliberately NOT launched.")
    print("Execution requires an authorized token (execution_authorized: true).")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
