"""Phase 13.10 pre-execution attestation and authorization gate tests."""
from dataclasses import replace

import pytest

from researchforge.benchmarks.rf_flex_confirmatory import RFFlexConfirmatoryProtocol
from researchforge.benchmarks.rf_flex_execution import (
    RFExecutionNotAuthorizedError,
    RFPreExecutionAttestation,
    assert_execution_ready,
    authorize_attestation,
    execute_authorized,
)


def _attestation(clean=True, authorized=False):
    value = RFPreExecutionAttestation(
        attestation_version="13.10.0",
        created_at="2026-09-06T00:00:00+00:00",
        phase13_9_protocol_fingerprint=RFFlexConfirmatoryProtocol().fingerprint(),
        protocol_file_sha256="protocol-file",
        task_manifest_fingerprint="task-manifest",
        software_commit="a" * 40,
        dirty_worktree=not clean,
        environment={"environment_fingerprint": "env"},
        planned_trajectory_groups=720,
        planned_conditions=8,
        planned_tasks=6,
        planned_seeds=5,
        planned_orderings=3,
        statistical_analysis_during_execution=False,
        authorization=authorized,
        authorized_by="reviewer" if authorized else None,
        authorization_timestamp="2026-09-06T00:00:00+00:00" if authorized else None,
        preflight_pass=clean,
    )
    return replace(value, attestation_fingerprint=value.fingerprint())


def test_unauthorized_execution_is_blocked():
    with pytest.raises(RFExecutionNotAuthorizedError):
        execute_authorized(_attestation(), RFFlexConfirmatoryProtocol(), lambda: "executed")


def test_dirty_worktree_cannot_be_authorized():
    with pytest.raises(Exception, match="clean-worktree"):
        authorize_attestation(_attestation(clean=False), "reviewer")


def test_authorization_changes_only_attestation_status_fields():
    initial = _attestation()
    authorized = authorize_attestation(initial, "reviewer")

    assert authorized.authorization is True
    assert authorized.phase13_9_protocol_fingerprint == initial.phase13_9_protocol_fingerprint
    assert authorized.task_manifest_fingerprint == initial.task_manifest_fingerprint
    assert authorized.software_commit == initial.software_commit
    assert authorized.planned_trajectory_groups == 720
    assert authorized.attestation_fingerprint != initial.attestation_fingerprint
    assert_execution_ready(authorized, RFFlexConfirmatoryProtocol())
    assert execute_authorized(authorized, RFFlexConfirmatoryProtocol(), lambda: "executed") == "executed"
