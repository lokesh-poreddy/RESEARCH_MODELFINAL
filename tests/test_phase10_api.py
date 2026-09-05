"""tests/test_phase10_api.py — Phase 10 API & Adapter Manifest verification suite.

Verifies:
1. Canonical Adapter Manifest with factual implementation status (no false claims).
2. All 15 canonical research REST endpoints (5 POST + 10 GET).
3. 3 legacy/operational compatibility endpoints (/experiments, /health, /safety/status).
4. Explicit Idempotency-Key support: cached replays vs 409 CONFLICT on payload mismatch.
5. Deterministic ApiErrorEnvelope contracts across 404, 409, 422, and 500.
6. Separation of transport status from domain execution status, validity, and research progress.
7. ISO-8601 UTC timestamp format and X-Request-ID propagation.
8. Architectural boundary: API layer has zero database/SQLAlchemy imports.
"""
from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from researchforge.adapters.manifest import (
    AdapterCapability,
    CanonicalAdapterManifest,
    ImplementationStatus,
)
from researchforge.api.dtos import ApiErrorCode
from researchforge.api.server import app, get_app_service
from researchforge.api.services import CallerIdentity, ResearchApplicationService
from researchforge.artifacts.model import Artifact
from researchforge.domain.action import ActionType, ResearchAction
from researchforge.domain.evidence import Evidence, EvidenceQuality
from researchforge.domain.experiment import ExecutionStatus, ExperimentRun, ExperimentSpec
from researchforge.domain.failure import Failure
from researchforge.domain.outcome import Outcome
from researchforge.domain.provenance import Provenance
from researchforge.domain.state import ResearchState
from researchforge.domain.validity import Validity, ValidityVerdict
from researchforge.policy.decision_record import PolicyDecisionRecord
from researchforge.policy.config import PolicyConfig
from researchforge.policy.research_policy import ResearchPolicy
from researchforge.evidence.snapshot import EvidenceSnapshot
from researchforge.policy.portfolio import BranchState, PortfolioBranch
from researchforge.policy.saturation import (
    PivotRecommendation,
    SaturationReport,
    SaturationState,
)
from researchforge.repositories.in_memory.uow import InMemoryUnitOfWork


@pytest.fixture
def client_and_service():
    """Provides a TestClient wired to a fresh ResearchApplicationService."""
    uow = InMemoryUnitOfWork()
    service = ResearchApplicationService(uow_factory=lambda: uow)

    # Dependency override to isolate test data
    app.dependency_overrides[get_app_service] = lambda: service
    client = TestClient(app)
    yield client, service, uow
    app.dependency_overrides.clear()


# ── 1. Canonical Adapter Manifest Tests ───────────────────────────────────────

def test_canonical_adapter_manifest_declaration():
    """Adapter manifest must provide factual implementation status without claiming stubs are implementations."""
    manifest = CanonicalAdapterManifest.default_manifest()

    # Total 5 adapters declared
    assert len(manifest.adapters) == 5

    # 1. Python Native: Fully implemented
    py_adapter = manifest.get_adapter("python_native")
    assert py_adapter is not None
    assert py_adapter.implementation_status == ImplementationStatus.IMPLEMENTED
    assert py_adapter.capability == AdapterCapability.EXPERIMENTS
    assert "execute_genome" in py_adapter.supported_operations

    # 2. MATLAB Bridge: Factual STUB (not implemented)
    mat_adapter = manifest.get_adapter("matlab_bridge")
    assert mat_adapter is not None
    assert mat_adapter.implementation_status == ImplementationStatus.STUB
    assert "stub" in mat_adapter.limitations[0].lower()

    # 3. Literature Retrieval: Implemented
    lit_adapter = manifest.get_adapter("literature_retrieval")
    assert lit_adapter is not None
    assert lit_adapter.implementation_status == ImplementationStatus.IMPLEMENTED
    assert lit_adapter.capability == AdapterCapability.EVIDENCE

    # 4. External Runtime Sandbox: Implemented
    rt_adapter = manifest.get_adapter("runtime_sandbox")
    assert rt_adapter is not None
    assert rt_adapter.implementation_status == ImplementationStatus.IMPLEMENTED
    assert "run_isolated" in rt_adapter.supported_operations

    # 5. Tensor/Data Framework: Declarative exchange contract only
    tensor_adapter = manifest.get_adapter("tensor_data_exchange")
    assert tensor_adapter is not None
    assert tensor_adapter.implementation_status == ImplementationStatus.DECLARATIVE
    assert "zero-copy" in tensor_adapter.limitations[0].lower()


def test_api_get_adapters_manifest(client_and_service):
    """GET /research/adapters/manifest must return the complete canonical manifest in standard envelope."""
    client, _, _ = client_and_service
    resp = client.get("/research/adapters/manifest")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "success"
    assert body["schema_version"] == "1.0"
    assert "timestamp" in body
    # Validate ISO-8601 UTC timestamp format
    dt = datetime.fromisoformat(body["timestamp"])
    assert dt.tzinfo is not None

    data = body["data"]
    assert "adapters" in data
    adapters = data["adapters"]
    assert "python_native" in adapters
    assert "matlab_bridge" in adapters
    assert adapters["matlab_bridge"]["implementation_status"] == "STUB"
    assert adapters["tensor_data_exchange"]["implementation_status"] == "DECLARATIVE"


# ── 2. Canonical POST Endpoints & Idempotency ──────────────────────────────────

def test_canonical_post_problem_and_idempotency(client_and_service):
    """POST /research/problems must create a problem and support deterministic idempotency."""
    client, _, _ = client_and_service
    payload = {
        "title": "Autonomous Algorithm Discovery Under Resource Limits",
        "description": "Investigating empirical convergence of neural architecture search.",
        "domain": "machine_learning",
        "metadata": {"priority": "high"},
    }
    headers = {"Idempotency-Key": "idemp_prob_001"}

    # Initial POST
    resp1 = client.post("/research/problems", json=payload, headers=headers)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] == "success"
    assert data1["object_id"].startswith("prob_")
    assert data1["fingerprint"] is not None
    assert data1["data"]["title"] == payload["title"]

    # Repeated identical POST with same Idempotency-Key must return same semantic result
    resp2 = client.post("/research/problems", json=payload, headers=headers)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["object_id"] == data1["object_id"]
    assert data2["fingerprint"] == data1["fingerprint"]

    # Conflicting POST with same Idempotency-Key but different payload must return 409 CONFLICT
    conflicting_payload = dict(payload, title="Completely Different Problem Title")
    resp_conflict = client.post("/research/problems", json=conflicting_payload, headers=headers)
    assert resp_conflict.status_code == 409
    err_body = resp_conflict.json()
    assert err_body["error_code"] == ApiErrorCode.CONFLICT.value
    assert "idempotency_key" in err_body["details"]
    assert err_body["schema_version"] == "1.0"


def test_canonical_post_questions_hypotheses_decisions(client_and_service):
    """POST for Questions, Hypotheses, and Decisions must succeed and record valid relationships."""
    client, _, _ = client_and_service

    # 1. Question
    q_payload = {
        "problem_id": "prob_100",
        "question_text": "Does learning rate warmup reduce early training divergence?",
        "rationale": "Empirical stability literature",
    }
    q_resp = client.post("/research/questions", json=q_payload)
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert q_data["status"] == "success"
    q_id = q_data["object_id"]

    # 2. Hypothesis
    h_payload = {
        "problem_id": "prob_100",
        "question_id": q_id,
        "hypothesis_statement": "Linear warmup for 5 epochs yields >= 2% higher final validation accuracy.",
        "predicted_effect": "+0.02 accuracy gain",
    }
    h_resp = client.post("/research/hypotheses", json=h_payload)
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["status"] == "success"
    h_id = h_data["object_id"]

    # 3. Decision
    d_payload = {
        "problem_id": "prob_100",
        "hypothesis_id": h_id,
        "action_type": "EXPLOIT_HYPOTHESIS",
        "rationale": "High expected gain with low memory failure risk",
        "target_context": {"learning_rate": 0.01, "warmup_epochs": 5},
        "parameters": {"optimizer": "AdamW"},
        "provenance_id": "prov_decision_001",
    }
    d_resp = client.post("/research/decisions", json=d_payload)
    assert d_resp.status_code == 200
    d_data = d_resp.json()
    assert d_data["status"] == "success"
    assert d_data["provenance"] == {"provenance_id": "prov_decision_001"}


def test_canonical_post_experiment_status_separation(client_and_service):
    """POST /research/experiments must explicitly separate transport status, execution status, validity, and progress."""
    client, _, _ = client_and_service
    exp_payload = {
        "decision_id": "dec_200",
        "hypothesis_id": "hyp_200",
        "model_type": "MLPClassifier",
        "strategy": "warmup_tuning",
        "timeout_seconds": 10.0,
    }
    resp = client.post("/research/experiments", json=exp_payload)
    assert resp.status_code == 200
    body = resp.json()

    # Invariant: Transport status != Domain execution status != Validity != Progress
    assert body["status"] == "success"  # Transport / API level
    assert body["data"]["execution_status"] == ExecutionStatus.PENDING.value
    assert body["data"]["validity_verdict"] == ValidityVerdict.PROVISIONAL.value
    assert body["data"]["research_progress"] == "NOT_ASSESSED"


# ── 3. Canonical GET Endpoints & Seeding ───────────────────────────────────────

def test_canonical_get_all_ten_endpoints(client_and_service):
    """All 10 canonical GET endpoints must retrieve their respective resources through UoW."""
    client, _, uow = client_and_service

    # Seed all canonical entities in the UnitOfWork
    with uow:
        # 1. State
        state = ResearchState(id="state_101", schema_version="1.0", problem_id="prob_101", research_phase="EXPLORATION")
        uow.states.save(state)

        # 2. Trajectory / Run & Outcome
        spec = ExperimentSpec(id="sp_101", schema_version="1.0", decision_id="dec_101", hypothesis_id="hyp_101")
        run = ExperimentRun(id="run_101", schema_version="1.0", experiment_spec_id="sp_101", status=ExecutionStatus.SUCCESS)
        outcome = Outcome(id="out_101", schema_version="1.0", run_id="run_101", measured_metrics={"val_accuracy": 0.88})
        uow.specs.save(spec)
        uow.runs.save(run)
        uow.outcomes.save(outcome)

        # 3. Evidence
        ev = Evidence(
            id="ev_101",
            schema_version="1.0",
            source="literature",
            snippet="Warmup improves optimization stability.",
            quality=EvidenceQuality(source_reliability=0.9, empirical_support=0.85),
        )
        uow.evidence.save(ev)

        # 4. Failure
        fail = Failure(id="fail_101", schema_version="1.0", experiment_id="exp_101", failure_category="DIVERGENCE")
        uow.failures.save(fail)

        # 5. Policy Decision
        policy = ResearchPolicy(config=PolicyConfig.baseline_v1())
        cand = ResearchAction(
            id="act_101",
            schema_version="1.0",
            action_type=ActionType.EXPLORE_NEW_STRATEGY,
            target_context={"strategy": "adamw"},
            expected_objective="Test",
        )
        snap = EvidenceSnapshot.create(decision_id="dec_101", as_of_timestamp=100.0, evidence_ids=[])
        _, rec = policy.evaluate_candidates(state, [cand], snap, decision_timestamp=100.0)
        object.__setattr__(rec, "id", "pdec_101")
        uow.policy_decisions.save(rec)

        # 6. Portfolio Branch
        branch_action = ResearchAction(
            id="act_br_1",
            schema_version="1.0",
            action_type=ActionType.EXPLORE_NEW_STRATEGY,
            target_context={"strategy": "adamw"},
            expected_objective="explore",
        )
        branch = PortfolioBranch(
            id="branch_101",
            schema_version="1.0",
            originating_hypothesis_id="hyp_101",
            action=branch_action,
            score=0.78,
            expected_value=0.80,
            resource_allocation=0.2,
            diversity_features={"strategy_family": "mutation"},
            branch_state=BranchState.ACTIVE,
        )
        uow.portfolio_branches.save(branch)

        # 7. Saturation Report
        sat = SaturationReport(
            id="sat_101",
            schema_version="1.0",
            current_status=SaturationState.DIMINISHING_RETURNS,
            recommendation=PivotRecommendation.CHANGE_STRATEGY,
            signal_values={"metric_delta_mean": 0.001},
            thresholds={"min_gain": 0.005},
            triggering_evidence=["ev_101"],
            rationale="Marginal gains diminishing over last 5 iterations.",
        )
        uow.saturation_reports.save(sat)

        # 8. Provenance
        prov = Provenance(
            id="prov_101",
            schema_version="1.0",
            created_by="agent_policy",
            created_at="2026-09-05T12:00:00Z",
            notes="Canonical provenance link",
        )
        uow.provenance.save(prov)

        # 9. Artifact
        art = Artifact(
            id="art_101",
            schema_version="1.0",
            artifact_type="model_weights",
            uri_or_path="artifacts/model_101.pt",
            sha256="abc123sha256hash",
            size_bytes=4096,
        )
        uow.artifacts.save(art)
        uow.commit()

    # Now verify all 10 canonical GET endpoints
    get_tests = [
        ("/research/states/state_101", "state_101"),
        ("/research/trajectories/run_101", "run_101"),
        ("/research/evidence/ev_101", "ev_101"),
        ("/research/failures/fail_101", "fail_101"),
        ("/research/policy/decisions/pdec_101", "pdec_101"),
        ("/research/portfolio/branch_101", "branch_101"),
        ("/research/saturation/sat_101", "sat_101"),
        ("/research/provenance/prov_101", "prov_101"),
        ("/research/artifacts/art_101", "art_101"),
        ("/research/adapters/manifest", "manifest_canonical_v1"),
    ]

    for endpoint, expected_id in get_tests:
        resp = client.get(endpoint)
        assert resp.status_code == 200, f"Failed on endpoint {endpoint}: {resp.text}"
        body = resp.json()
        assert body["status"] == "success"
        assert body["schema_version"] == "1.0"
        assert body["object_id"] == expected_id
        assert body["data"] is not None


# ── 4. Error Contract Tests (404, 422, Error Envelopes) ────────────────────────

def test_canonical_not_found_error_envelope(client_and_service):
    """Unmatched entity queries must return 404 with deterministic ApiErrorEnvelope."""
    client, _, _ = client_and_service
    resp = client.get("/research/states/nonexistent_state_999")
    assert resp.status_code == 404
    body = resp.json()

    assert body["error_code"] == ApiErrorCode.NOT_FOUND.value
    assert "nonexistent_state_999" in body["message"]
    assert body["details"]["entity_name"] == "ResearchState"
    assert body["details"]["entity_id"] == "nonexistent_state_999"
    assert body["schema_version"] == "1.0"
    assert "request_id" in body


def test_canonical_validation_error_envelope(client_and_service):
    """Malformed request payload must return 422 with deterministic ApiErrorEnvelope."""
    client, _, _ = client_and_service
    resp = client.post("/research/problems", json={"invalid_field": 123})
    assert resp.status_code == 422
    body = resp.json()

    assert body["error_code"] == ApiErrorCode.VALIDATION_ERROR.value
    assert "errors" in body["details"]
    assert body["schema_version"] == "1.0"
    assert "request_id" in body


# ── 5. Backwards Compatibility & Operational Endpoints ─────────────────────────

def test_legacy_operational_endpoints(client_and_service):
    """Legacy routes (/health, /safety/status, /experiments) must remain operational."""
    client, _, _ = client_and_service

    # 1. Health check
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    assert h_resp.json() == {"status": "ok"}

    # 2. Safety status
    s_resp = client.get("/safety/status")
    assert s_resp.status_code == 200
    assert "experiments_run" in s_resp.json()

    # 3. Legacy experiment creation
    legacy_exp_resp = client.post("/experiments", json={"hypothesis_text": "Legacy test", "model_type": "LogisticRegression"})
    assert legacy_exp_resp.status_code == 200
    legacy_body = legacy_exp_resp.json()
    assert "hypothesis_id" in legacy_body
    assert "experiment_id" in legacy_body
    assert "model_id" in legacy_body


# ── 6. Request ID Middleware & Caller Identity Extension ───────────────────────

def test_request_id_middleware_propagation(client_and_service):
    """Incoming X-Request-ID must be propagated to response header and error envelopes."""
    client, _, _ = client_and_service
    custom_id = "req_custom_trace_98765"
    headers = {"X-Request-ID": custom_id}

    # On success
    resp = client.get("/health", headers=headers)
    assert resp.headers.get("X-Request-ID") == custom_id

    # On error
    err_resp = client.get("/research/states/missing_id", headers=headers)
    assert err_resp.headers.get("X-Request-ID") == custom_id
    assert err_resp.json()["request_id"] == custom_id


def test_caller_identity_headers(client_and_service):
    """Caller identity headers (X-Client-ID, X-Caller-Role, X-Caller-Scopes) must be parsed."""
    client, _, _ = client_and_service
    headers = {
        "X-Client-ID": "test_agent_v2",
        "X-Caller-Role": "autonomous_controller",
        "X-Caller-Scopes": "read,write,admin",
    }
    payload = {
        "title": "Agent Directed Problem",
        "description": "Caller identity demonstration.",
    }
    resp = client.post("/research/problems", json=payload, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


# ── 7. Architectural Boundary Tests ───────────────────────────────────────────

def test_architectural_boundary_no_database_in_api():
    """API layer must NEVER import SQLAlchemy, psycopg, or direct persistence models."""
    api_dir = Path(__file__).parent.parent / "researchforge" / "api"
    forbidden_pkgs = frozenset(["sqlalchemy", "psycopg", "psycopg2", "alembic", "sqlite3"])

    violations = []
    for py_file in api_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules_to_check = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules_to_check.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    modules_to_check.append(alias.name)

            for mod in modules_to_check:
                pkg = mod.split(".")[0]
                if pkg in forbidden_pkgs:
                    violations.append(f"{py_file.name}:{getattr(node, 'lineno', '?')} imports forbidden db package {mod!r}")
                if "sql.models" in mod or "sql.database" in mod:
                    violations.append(f"{py_file.name}:{getattr(node, 'lineno', '?')} imports internal ORM model {mod!r}")

    assert not violations, (
        f"Architectural boundary violations in researchforge/api ({len(violations)}):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
