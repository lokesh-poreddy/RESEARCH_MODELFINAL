"""tests/test_phase13_boundary_and_hash_immutability.py — Phase 12B sealed hash immutability test.

Classification: CORE
Verifies Criterion 8: All sealed Phase 12B artifact physical and canonical hashes
remain bitwise identical before and after Phase 13 implementation.
"""
import json
from pathlib import Path
import pytest

from researchforge.benchmarks.cohort.specification import create_canonical_cohort_spec
from researchforge.domain.base import compute_canonical_fingerprint, compute_file_sha256

PROJECT_ROOT = Path(__file__).parent.parent

# Exact sealed Phase 12B physical hashes
SEALED_PHYSICAL_HASHES = {
    "phase12b_cohort_specification.json": "7869501b391b29a74664c1f841c91eb789a30219be5d96973b37f7ea899e142e",
    "phase12b_execution_manifest.json": "299654220fdc9bb850f15c636d410d77803c341dee07795ba36f452027366ec4",
    "phase12b_preflight_report.json": "058126d20434fa94a0c5006008e62ca01a96f31a961262c0b27902ff07197bb5",
    "phase12b_preflight_attestation.json": "e7a097ad494971673464d974c9da6af469af6807959fb9d37724df98a0ff7bc6",
    "phase12b_artifact_registry.json": "f2f899a1c9de17fa36c0c1c4c77e40ff2aa62d470b7297d5ebb9316d920309be",
}

# Exact sealed Phase 12B canonical fingerprints
SEALED_CANONICAL_FINGERPRINTS = {
    "phase12b_cohort_specification.json": "591e1a0bf62d9f0e9bfe7d717b66582dd89b1c7cc65d45082ff26607bc031e66",
    "phase12b_execution_manifest.json": "30142af747617e443c2c4afd43d7fcdce9603b5fe9a8b00c0b14c1902ac8969d",
    "phase12b_preflight_report.json": "cfebd3caeea9b5e13a1743ba0b4025766f118c55ca85a2587f6960ee62c1fdb3",
    "phase12b_preflight_attestation.json": "1658d1d9ba0d2c2755fe6e7081f09f140aa04b97171b020386a7adf96be152e6",
}


def test_phase12b_physical_file_hashes_immutable():
    """Verify all 5 sealed Phase 12B physical files match their exact sealed byte SHA-256 hashes."""
    for filename, expected_hash in SEALED_PHYSICAL_HASHES.items():
        file_path = PROJECT_ROOT / filename
        assert file_path.exists(), f"Sealed artifact {filename} missing from repository!"
        actual_hash = compute_file_sha256(file_path)
        assert actual_hash == expected_hash, (
            f"Physical file tampering detected in sealed artifact '{filename}'!\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )


def test_phase12b_canonical_fingerprints_immutable():
    """Verify all sealed Phase 12B canonical object fingerprints remain bitwise identical."""
    # 1. Cohort
    cohort_path = PROJECT_ROOT / "phase12b_cohort_specification.json"
    with open(cohort_path, "r", encoding="utf-8") as f:
        cohort_dict = json.load(f)
    assert cohort_dict["cohort_fingerprint"] == SEALED_CANONICAL_FINGERPRINTS["phase12b_cohort_specification.json"]
    fresh_cohort = create_canonical_cohort_spec()
    assert fresh_cohort.cohort_fingerprint == SEALED_CANONICAL_FINGERPRINTS["phase12b_cohort_specification.json"]

    # 2. Manifest
    manifest_path = PROJECT_ROOT / "phase12b_execution_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_dict = json.load(f)
    assert manifest_dict["manifest_fingerprint"] == SEALED_CANONICAL_FINGERPRINTS["phase12b_execution_manifest.json"]

    # 3. Preflight Report
    report_path = PROJECT_ROOT / "phase12b_preflight_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report_dict = json.load(f)
    assert report_dict["report_fingerprint"] == SEALED_CANONICAL_FINGERPRINTS["phase12b_preflight_report.json"]
    assert compute_canonical_fingerprint(report_dict, ["report_fingerprint"]) == SEALED_CANONICAL_FINGERPRINTS["phase12b_preflight_report.json"]

    # 4. Attestation
    attestation_path = PROJECT_ROOT / "phase12b_preflight_attestation.json"
    with open(attestation_path, "r", encoding="utf-8") as f:
        attestation_dict = json.load(f)
    assert attestation_dict["attestation_fingerprint"] == SEALED_CANONICAL_FINGERPRINTS["phase12b_preflight_attestation.json"]
    assert compute_canonical_fingerprint(attestation_dict, ["attestation_fingerprint"]) == SEALED_CANONICAL_FINGERPRINTS["phase12b_preflight_attestation.json"]
