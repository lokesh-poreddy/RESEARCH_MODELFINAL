"""tests/test_phase12b_execution_protocol.py — Unit tests for Phase 12B.2 Execution Protocol & Preflight Gates.

Verifies:
1. Manifest structure, catalog completeness, and deterministic fingerprinting (540 entries, 90 groups, 32,400 trials)
2. Two-tier transfer regime separation (sequence_transfer_regime vs condition_effective_regime)
3. Causal DAG dependencies and topological ordering (zero cycles, strict intra-group predecessor relations)
4. Bitwise-identical task materialization and canonical byte-level array hashing
5. Exhaustive 8-point Preflight Verification Gate suite execution
6. Preflight attestation token generation, dual-domain binding, and cryptographic integrity
7. Human authorization safety gate (machine verification != authorization; ExecutionNotAuthorizedError raised)
8. Complete cryptographic chain validation (C <-> SAP <-> M <-> P <-> A)
9. Canonical vs physical hash distinction (pretty-print invariant to canonical hash, sensitive to physical hash)
10. Anti-circularity self-hash exclusion
11. Physical file tampering & in-memory substitution detection
12. Authorization invalidation after artifact modification
13. Immutable 40-character Git commit SHA binding
14. Ledger appending and auditable artifact registry tracking
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import pytest

from researchforge.domain.base import (
    compute_canonical_fingerprint,
    compute_file_sha256,
)
from researchforge.benchmarks.cohort import (
    BenchmarkCondition,
    TaskFamily,
    TransferRegime,
    create_canonical_cohort_spec,
)
from researchforge.benchmarks.execution import (
    ArtifactRegistryRecord,
    BenchmarkExecutionProtocol,
    BenchmarkTaskMaterialization,
    ExecutionManifest,
    ExecutionNotAuthorizedError,
    LedgerEntry,
    PreflightAttestation,
    PreflightCheckResult,
    PreflightGateError,
    PreflightValidationReport,
    PreflightValidator,
    RunManifestEntry,
    compute_canonical_task_content_hash,
    generate_execution_manifest,
    get_software_commit_info,
    materialize_benchmark_task,
)
from researchforge.benchmarks.statistical import create_canonical_sap


@pytest.fixture
def canonical_manifest() -> ExecutionManifest:
    cohort = create_canonical_cohort_spec()
    sap = create_canonical_sap()
    return generate_execution_manifest(cohort, sap)


@pytest.fixture
def preflight_validator() -> PreflightValidator:
    cohort = create_canonical_cohort_spec()
    sap = create_canonical_sap()
    return PreflightValidator(cohort, sap)


# ── 1. Manifest Structure & Catalog Completeness ─────────────────────────────────

def test_manifest_structure_and_counts(canonical_manifest: ExecutionManifest):
    """Manifest must contain exactly 540 entries, 90 groups, 32,400 trials, and valid fingerprints."""
    m = canonical_manifest
    assert m.entry_count == 540
    assert len(m.entries) == 540
    assert m.sequence_group_count == 90
    assert m.planned_trials == 32400
    assert m.cohort_version == "1.0.1"
    assert m.cohort_fingerprint == "591e1a0bf62d9f0e9bfe7d717b66582dd89b1c7cc65d45082ff26607bc031e66"
    assert m.sap_version == "1.0.0"
    assert m.manifest_fingerprint != ""
    assert len(m.manifest_fingerprint) == 64
    assert len(m.git_commit) == 40

    for e in m.entries:
        assert e.entry_fingerprint != ""
        assert len(e.entry_fingerprint) == 64
        assert e.entry_fingerprint == e.compute_fingerprint()


# ── 2. Two-Tier Transfer Regimes Separation ──────────────────────────────────────

def test_two_tier_regimes_separation(canonical_manifest: ExecutionManifest):
    """Sequence transfer regime must be decoupled from condition effective regime."""
    for e in canonical_manifest.entries:
        if e.sequence_position == 0:
            assert e.sequence_transfer_regime == TransferRegime.NO_PRIOR_EXPERIENCE
            assert e.condition_effective_regime == TransferRegime.NO_PRIOR_EXPERIENCE

        if e.condition in (BenchmarkCondition.COLD_START, BenchmarkCondition.NO_MEMORY):
            assert e.condition_effective_regime == TransferRegime.NO_PRIOR_EXPERIENCE
        elif e.sequence_position > 0:
            assert e.condition_effective_regime == e.sequence_transfer_regime

    forward_entries = [e for e in canonical_manifest.entries if e.ordering == "forward" and e.seed == 0 and e.condition == BenchmarkCondition.CONTINUOUS_EXPERIENCE]
    assert forward_entries[0].sequence_position == 0
    assert forward_entries[0].task_id == "digits_0_4"
    assert forward_entries[0].sequence_transfer_regime == TransferRegime.NO_PRIOR_EXPERIENCE
    assert forward_entries[0].condition_effective_regime == TransferRegime.NO_PRIOR_EXPERIENCE

    assert forward_entries[1].sequence_position == 1
    assert forward_entries[1].task_id == "digits_5_9"
    assert forward_entries[1].sequence_transfer_regime == TransferRegime.SAME_FAMILY
    assert forward_entries[1].condition_effective_regime == TransferRegime.SAME_FAMILY

    cross_entries = [e for e in canonical_manifest.entries if e.ordering == "cross_first" and e.seed == 0 and e.condition == BenchmarkCondition.CONTINUOUS_EXPERIENCE]
    assert cross_entries[0].sequence_position == 0
    assert cross_entries[0].task_id == "synthetic_ecg_lead1"
    assert cross_entries[1].sequence_position == 1
    assert cross_entries[1].task_id == "digits_0_4"
    assert cross_entries[1].sequence_transfer_regime == TransferRegime.CROSS_FAMILY
    assert cross_entries[1].condition_effective_regime == TransferRegime.CROSS_FAMILY


# ── 3. Causal DAG Dependencies & Topological Order ───────────────────────────────

def test_causal_dag_dependencies(canonical_manifest: ExecutionManifest):
    """Dependencies must be strictly intra-group predecessors with zero cycles."""
    entry_map = {e.entry_id: e for e in canonical_manifest.entries}
    assert len(entry_map) == 540

    for e in canonical_manifest.entries:
        if e.sequence_position == 0:
            assert len(e.depends_on_entry_ids) == 0
            assert len(e.prior_task_ids) == 0
        else:
            assert len(e.depends_on_entry_ids) == e.sequence_position
            assert len(e.prior_task_ids) == e.sequence_position
            for dep_id in e.depends_on_entry_ids:
                dep_entry = entry_map[dep_id]
                assert dep_entry.sequence_group_id == e.sequence_group_id
                assert dep_entry.sequence_position < e.sequence_position


# ── 4. Task Materialization Parity ───────────────────────────────────────────────

def test_task_materialization_parity():
    """Double materialization of all 6 tasks must produce bitwise identical arrays and canonical hashes."""
    task_ids = [
        "digits_0_4",
        "digits_5_9",
        "digits_all_10",
        "synthetic_ecg_lead1",
        "synthetic_ecg_lead2",
        "xor_parity_8bit",
    ]
    for tid in task_ids:
        m1 = materialize_benchmark_task(tid, seed=0)
        m2 = materialize_benchmark_task(tid, seed=0)

        h1 = compute_canonical_task_content_hash(m1.task)
        h2 = compute_canonical_task_content_hash(m2.task)
        assert h1 == h2
        assert len(h1) == 64
        assert m1.content_hash == m2.content_hash
        assert m1.task_id == tid


# ── 5. 8-Point Preflight Verification Gate Suite ─────────────────────────────────

def test_preflight_validator_all_8_gates(preflight_validator: PreflightValidator, canonical_manifest: ExecutionManifest):
    """All 8 preflight gates must pass without errors."""
    report = preflight_validator.validate_all(canonical_manifest)
    assert report.all_gates_passed is True
    assert len(report.gates) == 8
    assert report.cohort_fingerprint == "591e1a0bf62d9f0e9bfe7d717b66582dd89b1c7cc65d45082ff26607bc031e66"
    assert report.manifest_fingerprint == canonical_manifest.manifest_fingerprint
    assert report.report_fingerprint != ""

    for g in report.gates:
        assert g.status == "PASSED", f"Gate {g.gate_number} ({g.gate_name}) failed: {g.details}"
        assert g.latency_ms >= 0.0


# ── 6. Attestation Dual Binding & Anti-Circularity ───────────────────────────────

def test_preflight_attestation_machine_vs_human_separation(preflight_validator: PreflightValidator, canonical_manifest: ExecutionManifest):
    """Default attestation has preflight_pass=True but execution_authorized=False, binding full 40-char git commit."""
    report = preflight_validator.validate_all(canonical_manifest)
    attestation = preflight_validator.generate_attestation(report)

    assert attestation.preflight_pass is True
    assert attestation.execution_authorized is False
    assert attestation.authorized_by is None
    assert attestation.authorization_timestamp is None
    assert attestation.manifest_canonical_fingerprint == canonical_manifest.manifest_fingerprint
    assert attestation.cohort_canonical_fingerprint == canonical_manifest.cohort_fingerprint
    assert len(attestation.software_commit) == 40
    assert attestation.attestation_fingerprint != ""
    assert attestation.attestation_fingerprint == attestation.compute_fingerprint()


def test_self_hash_exclusion_prevents_circularity():
    """Attestation and manifest compute_fingerprint must exclude self-referential fields."""
    cohort = create_canonical_cohort_spec()
    sap = create_canonical_sap()
    manifest = generate_execution_manifest(cohort, sap)
    validator = PreflightValidator(cohort, sap)
    report = validator.validate_all(manifest)
    attestation = validator.generate_attestation(report)

    # Recomputing fingerprint with dummy or empty attestation_fingerprint must produce identical result
    fp1 = attestation.compute_fingerprint()
    attestation_mutated = PreflightAttestation(
        **{**attestation.__dict__, "attestation_fingerprint": "fake_fingerprint_000000000000000000000000000000000000000000000"}
    )
    fp2 = attestation_mutated.compute_fingerprint()
    assert fp1 == fp2, "Fingerprint must exclude self-referential attestation_fingerprint!"


def test_canonical_vs_physical_hash_distinction():
    """Pretty-printed JSON formatting changes physical file SHA-256 while leaving canonical fingerprint invariant."""
    sample_data = {"z_key": 100, "a_key": "alpha", "nested": [1, 2, 3]}

    with tempfile.TemporaryDirectory() as tmpdir:
        compact_file = Path(tmpdir) / "compact.json"
        pretty_file = Path(tmpdir) / "pretty.json"

        # Write compact JSON
        compact_file.write_text(json.dumps(sample_data, separators=(",", ":")), encoding="utf-8")
        # Write pretty-printed JSON with extra whitespace and indentation
        pretty_file.write_text(json.dumps(sample_data, indent=4), encoding="utf-8")

        # Physical hashes MUST differ due to whitespace bytes
        compact_phys = compute_file_sha256(compact_file)
        pretty_phys = compute_file_sha256(pretty_file)
        assert compact_phys != pretty_phys, "Physical file SHA-256 must detect whitespace differences!"

        # Canonical object fingerprints MUST be identical because semantic payload is unchanged
        compact_canon = compute_canonical_fingerprint(json.loads(compact_file.read_text(encoding="utf-8")))
        pretty_canon = compute_canonical_fingerprint(json.loads(pretty_file.read_text(encoding="utf-8")))
        assert compact_canon == pretty_canon, "Canonical fingerprint must be invariant to whitespace/formatting!"


# ── 7. Human Authorization Safety Gate & Chain Verification ─────────────────────

def test_safety_gate_blocks_unauthorized_execution(canonical_manifest: ExecutionManifest):
    """Protocol must refuse to execute benchmark without explicit human authorization."""
    proto = BenchmarkExecutionProtocol()
    report, attestation = proto.run_preflight(canonical_manifest, execution_authorized=False)

    assert attestation.preflight_pass is True
    assert attestation.execution_authorized is False

    with pytest.raises(ExecutionNotAuthorizedError) as exc:
        proto.execute_benchmark(canonical_manifest, attestation)
    assert "execution is NOT authorized" in str(exc.value)


def test_safety_gate_blocks_mismatched_manifest(canonical_manifest: ExecutionManifest):
    """Protocol must block execution if manifest canonical fingerprint does not match attestation."""
    proto = BenchmarkExecutionProtocol()
    report, attestation = proto.run_preflight(canonical_manifest, execution_authorized=True, authorized_by="TestAuthorizer")

    # In-memory substitution with alternate manifest
    tampered_manifest = ExecutionManifest(
        manifest_version="1.0.0",
        cohort_version="1.0.1",
        cohort_fingerprint=canonical_manifest.cohort_fingerprint,
        sap_version="1.0.0",
        sap_fingerprint=canonical_manifest.sap_fingerprint,
        manifest_fingerprint="tampered_canonical_fingerprint" + "0" * 32,
        entries=canonical_manifest.entries,
    )

    with pytest.raises(ExecutionNotAuthorizedError) as exc:
        proto.execute_benchmark(tampered_manifest, attestation)
    assert ("does not match loaded manifest" in str(exc.value) or
            "Manifest internal canonical fingerprint mismatch" in str(exc.value))


def test_physical_file_tampering_detection(canonical_manifest: ExecutionManifest):
    """Modifying physical bytes of manifest on disk must be detected and rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file = Path(tmpdir) / "phase12b_execution_manifest.json"
        manifest_file.write_text(json.dumps(canonical_manifest.to_dict(), indent=2), encoding="utf-8")

        proto = BenchmarkExecutionProtocol()
        report, attestation = proto.run_preflight(
            canonical_manifest,
            manifest_file_path=manifest_file,
            execution_authorized=True,
            authorized_by="LeadScientist",
        )
        assert attestation.manifest_physical_file_sha256 == compute_file_sha256(manifest_file)

        # Alter 1 byte on disk (append newline)
        with open(manifest_file, "a", encoding="utf-8") as f:
            f.write("\n")

        with pytest.raises(ExecutionNotAuthorizedError) as exc:
            proto.verify_authorization(
                manifest=canonical_manifest,
                attestation=attestation,
                manifest_file_path=manifest_file,
            )
        assert "Manifest physical file SHA-256 mismatch" in str(exc.value)


def test_attestation_signature_tamper_detection(canonical_manifest: ExecutionManifest):
    """Manually tampering with attestation fields without recomputing signature must be rejected."""
    proto = BenchmarkExecutionProtocol()
    report, attestation = proto.run_preflight(canonical_manifest, execution_authorized=False)

    # Attacker mutates execution_authorized to True without signature update
    tampered_attestation = PreflightAttestation(
        **{**attestation.__dict__, "execution_authorized": True}
    )

    with pytest.raises(ExecutionNotAuthorizedError) as exc:
        proto.verify_authorization(canonical_manifest, tampered_attestation)
    assert "Attestation cryptographic signature mismatch: token has been tampered with" in str(exc.value)


def test_git_commit_sha_mismatch_detection(canonical_manifest: ExecutionManifest):
    """Attestation bound to a different Git commit SHA must be rejected."""
    proto = BenchmarkExecutionProtocol()
    report, attestation = proto.run_preflight(
        canonical_manifest,
        software_commit="0123456789abcdef0123456789abcdef01234567",
        execution_authorized=True,
        authorized_by="LeadScientist",
    )

    with pytest.raises(ExecutionNotAuthorizedError) as exc:
        proto.verify_authorization(canonical_manifest, attestation, verify_git_commit=True)
    assert "Software commit SHA mismatch" in str(exc.value)


# ── 8. Ledger Appending & Artifact Registry ─────────────────────────────────────

def test_ledger_append_and_registry(canonical_manifest: ExecutionManifest):
    """Single task micro-execution must append valid JSONL ledger record and update artifact registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "phase12b_execution_ledger.jsonl"
        registry_path = Path(tmpdir) / "phase12b_artifact_registry.json"

        proto = BenchmarkExecutionProtocol(ledger_path=ledger_path, registry_path=registry_path)

        entry = canonical_manifest.entries[0]
        entry_micro = RunManifestEntry(
            entry_id=entry.entry_id,
            condition=entry.condition,
            seed=entry.seed,
            ordering=entry.ordering,
            sequence_group_id=entry.sequence_group_id,
            sequence_position=entry.sequence_position,
            task_id=entry.task_id,
            task_family=entry.task_family,
            task_fingerprint=entry.task_fingerprint,
            prior_task_ids=entry.prior_task_ids,
            prior_task_families=entry.prior_task_families,
            sequence_transfer_regime=entry.sequence_transfer_regime,
            condition_effective_regime=entry.condition_effective_regime,
            depends_on_entry_ids=entry.depends_on_entry_ids,
            generation_budget=1,
            population_size=2,
        )

        ledger_rec, next_state = proto.execute_task_entry(entry_micro)
        assert ledger_rec.entry_id == entry.entry_id
        assert ledger_rec.status in ("VALID_COMPLETED", "SCIENTIFIC_FAILURE")
        assert ledger_rec.entry_hash != ""

        proto.append_ledger_entry(ledger_rec)
        assert ledger_path.exists()
        lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["entry_id"] == entry.entry_id

        # Artifact registry test with dual-domain hashes
        dummy_art = Path(tmpdir) / "dummy_spec.json"
        dummy_art.write_text('{"key": "value"}', encoding="utf-8")
        rec = proto.register_artifact(
            artifact_name="dummy_spec",
            file_path=dummy_art,
            description="Dummy specification for test",
            canonical_fingerprint="dummy_canonical_123",
        )
        assert rec.artifact_name == "dummy_spec"
        assert rec.canonical_fingerprint == "dummy_canonical_123"
        assert rec.physical_file_sha256 == compute_file_sha256(dummy_art)
        assert rec.sha256_hash == rec.physical_file_sha256
        assert registry_path.exists()

        reg_data = json.loads(registry_path.read_text(encoding="utf-8"))
        assert len(reg_data["artifacts"]) == 1
        entry_reg = reg_data["artifacts"][0]
        assert entry_reg["artifact_name"] == "dummy_spec"
        assert entry_reg["canonical_fingerprint"] == "dummy_canonical_123"
        assert entry_reg["physical_file_sha256"] == rec.physical_file_sha256


# ── 9. Authorization Invalidation After Artifact Replacement ────────────────────

def test_authorization_invalidation_after_artifact_replacement(canonical_manifest: ExecutionManifest):
    """Replacing manifest A with manifest B invalidates authorization even if execution_authorized is True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file_a = Path(tmpdir) / "manifest_a.json"
        manifest_file_a.write_text(json.dumps(canonical_manifest.to_dict(), indent=2), encoding="utf-8")

        proto = BenchmarkExecutionProtocol()
        report_a, attestation_a = proto.run_preflight(
            canonical_manifest,
            manifest_file_path=manifest_file_a,
            execution_authorized=True,
            authorized_by="LeadScientist",
        )

        # Baseline: Manifest A is authorized and validates
        proto.verify_authorization(
            manifest=canonical_manifest,
            attestation=attestation_a,
            manifest_file_path=manifest_file_a,
        )

        # Manifest B: mutate a single entry in manifest and recompute its canonical fingerprint
        entries_b = list(canonical_manifest.entries)
        mutated_entry = RunManifestEntry(
            **{**entries_b[0].__dict__, "generation_budget": 99}
        )
        mutated_entry = RunManifestEntry(
            **{**mutated_entry.__dict__, "entry_fingerprint": mutated_entry.compute_fingerprint()}
        )
        entries_b[0] = mutated_entry
        manifest_b = ExecutionManifest(
            **{**canonical_manifest.__dict__, "entries": tuple(entries_b)}
        )
        manifest_b = ExecutionManifest(
            **{**manifest_b.__dict__, "manifest_fingerprint": manifest_b.compute_fingerprint()}
        )
        manifest_file_b = Path(tmpdir) / "manifest_b.json"
        manifest_file_b.write_text(json.dumps(manifest_b.to_dict(), indent=2), encoding="utf-8")

        # Case 1: Attestation A presented with Manifest B on disk -> Physical SHA-256 mismatch
        with pytest.raises(ExecutionNotAuthorizedError) as exc:
            proto.verify_authorization(
                manifest=canonical_manifest,
                attestation=attestation_a,
                manifest_file_path=manifest_file_b,
            )
        assert "Manifest physical file SHA-256 mismatch" in str(exc.value)

        # Case 2: Attestation A presented with Manifest B in-memory -> Canonical fingerprint mismatch
        with pytest.raises(ExecutionNotAuthorizedError) as exc:
            proto.verify_authorization(
                manifest=manifest_b,
                attestation=attestation_a,
                manifest_file_path=manifest_file_a,
            )
        assert "does not match loaded manifest" in str(exc.value)


# ── 10. Registry Non-Circularity ────────────────────────────────────────────────

def test_registry_non_circularity(canonical_manifest: ExecutionManifest):
    """Artifact registry is an index downstream of sealed artifacts and cannot affect attestation."""
    cohort = create_canonical_cohort_spec()
    sap = create_canonical_sap()
    validator = PreflightValidator(cohort, sap)
    report = validator.validate_all(canonical_manifest)
    attestation = validator.generate_attestation(report)

    # Attestation fingerprint is fixed
    fp_before = attestation.attestation_fingerprint

    # Mutating or creating an artifact registry downstream has ZERO effect on attestation signature
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = Path(tmpdir) / "registry.json"
        proto = BenchmarkExecutionProtocol(validator=validator, registry_path=reg_path)
        dummy_file = Path(tmpdir) / "test.json"
        dummy_file.write_text('{"item": 1}', encoding="utf-8")

        proto.register_artifact("dummy", dummy_file, "Dummy test", canonical_fingerprint="fp1")
        proto.register_artifact("dummy2", dummy_file, "Dummy test 2", canonical_fingerprint="fp2")

        # Attestation fingerprint remains completely invariant
        assert attestation.attestation_fingerprint == fp_before
        assert attestation.compute_fingerprint() == fp_before


# ── 11. Complete Chain Verification on Sealed On-Disk Artifacts ──────────────────

def test_on_disk_sealed_artifacts_complete_chain():
    """Validates the frozen Phase 12B.2 on-disk artifacts against complete authorization chain."""
    root = Path(__file__).parent.parent
    cohort_path = root / "phase12b_cohort_specification.json"
    manifest_path = root / "phase12b_execution_manifest.json"
    report_path = root / "phase12b_preflight_report.json"
    attestation_path = root / "phase12b_preflight_attestation.json"
    registry_path = root / "phase12b_artifact_registry.json"

    assert cohort_path.exists(), "Cohort specification must exist on disk"
    assert manifest_path.exists(), "Manifest file must exist on disk"
    assert report_path.exists(), "Preflight report must exist on disk"
    assert attestation_path.exists(), "Attestation file must exist on disk"
    assert registry_path.exists(), "Artifact registry must exist on disk"

    with open(manifest_path, encoding="utf-8") as f:
        m_dict = json.load(f)
    entries = [
        RunManifestEntry(
            **{
                **e,
                "prior_task_ids": tuple(e["prior_task_ids"]),
                "prior_task_families": tuple(TaskFamily(f) for f in e["prior_task_families"]),
                "depends_on_entry_ids": tuple(e["depends_on_entry_ids"]),
                "condition": BenchmarkCondition(e["condition"]),
                "task_family": TaskFamily(e["task_family"]),
                "sequence_transfer_regime": TransferRegime(e["sequence_transfer_regime"]),
                "condition_effective_regime": TransferRegime(e["condition_effective_regime"]),
            }
        )
        for e in m_dict["entries"]
    ]
    manifest = ExecutionManifest(**{**m_dict, "entries": tuple(entries)})

    with open(report_path, encoding="utf-8") as f:
        r_dict = json.load(f)
    report = PreflightValidationReport(**{**r_dict, "gates": ()})

    with open(attestation_path, encoding="utf-8") as f:
        a_dict = json.load(f)
    attestation = PreflightAttestation.from_dict(a_dict)

    proto = BenchmarkExecutionProtocol()

    # The chain MUST pass all cryptographic checks and fail ONLY at the human authorization safety lock
    with pytest.raises(ExecutionNotAuthorizedError) as exc:
        proto.verify_authorization(
            manifest=manifest,
            attestation=attestation,
            manifest_file_path=manifest_path,
            cohort_file_path=cohort_path,
            report_file_path=report_path,
            report=report,
            verify_git_commit=True,
        )
    assert "execution is NOT authorized" in str(exc.value)
    assert "Human authorization token is required before committing 32,400-trial budget" in str(exc.value)

