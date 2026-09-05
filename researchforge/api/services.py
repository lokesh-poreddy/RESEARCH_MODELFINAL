"""researchforge/api/services.py — Canonical ResearchForge Application Service.

Phase 10 Architecture Invariant:
    API (HTTP Controllers)
      ↓
    ResearchApplicationService
      ↓
    Domain Contracts & Base Objects
      ↓
    UnitOfWork Protocols & Repositories

API controllers MUST NEVER:
- Import SQLAlchemy models
- Open direct database connections
- Access or manipulate repository tables directly
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..adapters.manifest import CanonicalAdapterManifest
from ..domain.base import _canonical_json
from ..domain.decision import Decision
from ..domain.experiment import ExecutionStatus, ExperimentRun, ExperimentSpec
from ..domain.hypothesis import Hypothesis
from ..domain.outcome import Outcome
from ..domain.problem import ResearchProblem
from ..domain.provenance import Provenance
from ..domain.question import ResearchQuestion, QuestionStatus
from ..domain.state import ResearchState
from ..domain.validity import ValidityVerdict
from ..policy.portfolio import PortfolioBranch, ResearchPortfolio
from ..policy.saturation import PivotRecommendation, SaturationReport, SaturationState
from ..repositories.in_memory.uow import InMemoryUnitOfWork
from ..repositories.interfaces import EntityImmutabilityError, UnitOfWork
from .dtos import (
    ApiErrorCode,
    ApiResponseEnvelope,
    CreateDecisionRequest,
    CreateExperimentRequest,
    CreateHypothesisRequest,
    CreateProblemRequest,
    CreateQuestionRequest,
)


class ApplicationServiceError(Exception):
    """Base exception for application service logic."""
    def __init__(self, code: ApiErrorCode, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class EntityNotFoundError(ApplicationServiceError):
    def __init__(self, entity_name: str, entity_id: str) -> None:
        super().__init__(
            ApiErrorCode.NOT_FOUND,
            f"{entity_name} with id '{entity_id}' was not found.",
            {"entity_name": entity_name, "entity_id": entity_id},
        )


class IdempotencyConflictError(ApplicationServiceError):
    def __init__(self, key: str) -> None:
        super().__init__(
            ApiErrorCode.CONFLICT,
            f"Idempotency-Key '{key}' was previously used with a different request payload.",
            {"idempotency_key": key},
        )


@dataclass(frozen=True)
class CallerIdentity:
    """Extension point for client identity and authorization context."""
    client_id: str = "anonymous"
    role: str = "researcher"
    scopes: tuple[str, ...] = ("read", "write")


@dataclass
class IdempotencyRecord:
    """Durable record of an idempotent request execution."""
    key: str
    request_fingerprint: str
    envelope_dict: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


class ResearchApplicationService:
    """Core application coordinator orchestrating domain operations through UnitOfWork."""

    def __init__(self, uow_factory: Optional[Callable[[], UnitOfWork]] = None) -> None:
        self.uow_factory = uow_factory or (lambda: InMemoryUnitOfWork())
        self._idempotency_store: Dict[str, IdempotencyRecord] = {}
        self._manifest = CanonicalAdapterManifest.default_manifest()

    def _check_idempotency(self, key: Optional[str], payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not key:
            return None
        req_fp = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if key in self._idempotency_store:
            rec = self._idempotency_store[key]
            if rec.request_fingerprint != req_fp:
                raise IdempotencyConflictError(key)
            return rec.envelope_dict
        return None

    def _record_idempotency(self, key: Optional[str], payload: Dict[str, Any], envelope: ApiResponseEnvelope) -> None:
        if not key:
            return
        req_fp = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        self._idempotency_store[key] = IdempotencyRecord(
            key=key,
            request_fingerprint=req_fp,
            envelope_dict=envelope.model_dump(),
        )

    # ── 1. Problems ───────────────────────────────────────────────────────────

    def create_problem(
        self,
        req: CreateProblemRequest,
        idempotency_key: Optional[str] = None,
        caller: Optional[CallerIdentity] = None,
    ) -> ApiResponseEnvelope:
        cached = self._check_idempotency(idempotency_key, req.model_dump())
        if cached:
            return ApiResponseEnvelope(**cached)

        prob_id = f"prob_{int(time.time() * 1000)}"
        meta = dict(req.metadata)
        meta["domain"] = req.domain

        prob = ResearchProblem(
            id=prob_id,
            schema_version="1.0",
            title=req.title,
            description=req.description,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=meta,
        )

        uow = self.uow_factory()
        with uow:
            uow.problems.save(prob)
            uow.commit()

        env = ApiResponseEnvelope(
            object_id=prob.id,
            status="success",
            fingerprint=prob.fingerprint(),
            data=prob.to_dict(),
        )
        self._record_idempotency(idempotency_key, req.model_dump(), env)
        return env

    # ── 2. Questions ──────────────────────────────────────────────────────────

    def create_question(
        self,
        req: CreateQuestionRequest,
        idempotency_key: Optional[str] = None,
        caller: Optional[CallerIdentity] = None,
    ) -> ApiResponseEnvelope:
        cached = self._check_idempotency(idempotency_key, req.model_dump())
        if cached:
            return ApiResponseEnvelope(**cached)

        q_id = f"q_{int(time.time() * 1000)}"
        q = ResearchQuestion(
            id=q_id,
            schema_version="1.0",
            problem_id=req.problem_id,
            question_text=req.question_text,
            rationale=req.rationale or "",
            status=QuestionStatus.OPEN,
        )

        uow = self.uow_factory()
        with uow:
            uow.questions.save(q)
            uow.commit()

        env = ApiResponseEnvelope(
            object_id=q.id,
            status="success",
            fingerprint=q.fingerprint(),
            data=q.to_dict(),
        )
        self._record_idempotency(idempotency_key, req.model_dump(), env)
        return env

    # ── 3. Hypotheses ─────────────────────────────────────────────────────────

    def create_hypothesis(
        self,
        req: CreateHypothesisRequest,
        idempotency_key: Optional[str] = None,
        caller: Optional[CallerIdentity] = None,
    ) -> ApiResponseEnvelope:
        cached = self._check_idempotency(idempotency_key, req.model_dump())
        if cached:
            return ApiResponseEnvelope(**cached)

        h_id = f"hyp_{int(time.time() * 1000)}"
        hyp = Hypothesis(
            id=h_id,
            schema_version="1.0",
            research_question_id=req.question_id,
            statement=req.hypothesis_statement,
            prediction=req.predicted_effect,
        )

        uow = self.uow_factory()
        with uow:
            uow.hypotheses.save(hyp)
            uow.commit()

        env = ApiResponseEnvelope(
            object_id=hyp.id,
            status="success",
            fingerprint=hyp.fingerprint(),
            data=hyp.to_dict(),
        )
        self._record_idempotency(idempotency_key, req.model_dump(), env)
        return env

    # ── 4. Decisions ──────────────────────────────────────────────────────────

    def create_decision(
        self,
        req: CreateDecisionRequest,
        idempotency_key: Optional[str] = None,
        caller: Optional[CallerIdentity] = None,
    ) -> ApiResponseEnvelope:
        cached = self._check_idempotency(idempotency_key, req.model_dump())
        if cached:
            return ApiResponseEnvelope(**cached)

        d_id = f"dec_{int(time.time() * 1000)}"
        dec = Decision(
            id=d_id,
            schema_version="1.0",
            research_state_fingerprint=req.target_context.get("state_fingerprint"),
            rsg_id=req.parameters.get("rsg_id"),
            hypothesis_id=req.hypothesis_id,
            selected_tmg_id=req.parameters.get("tmg_id"),
            selected_operator=req.action_type,
            decision_reason=req.rationale,
            decision_timestamp=datetime.now(timezone.utc).isoformat(),
            policy_version="1.0.0",
        )

        uow = self.uow_factory()
        with uow:
            uow.decisions.save(dec)
            uow.commit()

        env = ApiResponseEnvelope(
            object_id=dec.id,
            status="success",
            fingerprint=dec.fingerprint(),
            provenance={"provenance_id": req.provenance_id} if req.provenance_id else None,
            data=dec.to_dict(),
        )
        self._record_idempotency(idempotency_key, req.model_dump(), env)
        return env

    # ── 5. Experiments ────────────────────────────────────────────────────────

    def create_experiment(
        self,
        req: CreateExperimentRequest,
        idempotency_key: Optional[str] = None,
        caller: Optional[CallerIdentity] = None,
    ) -> ApiResponseEnvelope:
        cached = self._check_idempotency(idempotency_key, req.model_dump())
        if cached:
            return ApiResponseEnvelope(**cached)

        sp_id = f"sp_{int(time.time() * 1000)}"
        spec = ExperimentSpec(
            id=sp_id,
            schema_version="1.0",
            decision_id=req.decision_id,
            hypothesis_id=req.hypothesis_id,
            target_model_genome_id=req.model_type,
            intervention_description=req.strategy,
            resource_requirements={"timeout_seconds": req.timeout_seconds},
        )

        run_id = f"run_{sp_id}"
        run = ExperimentRun(
            id=run_id,
            schema_version="1.0",
            experiment_spec_id=spec.id,
            status=ExecutionStatus.PENDING,
        )

        uow = self.uow_factory()
        with uow:
            uow.specs.save(spec)
            uow.runs.save(run)
            uow.commit()

        # Scientific validity vs execution status distinction explicitly exposed
        env = ApiResponseEnvelope(
            object_id=spec.id,
            status="success",
            fingerprint=spec.fingerprint(),
            data={
                "experiment_spec": spec.to_dict(),
                "experiment_run": run.to_dict(),
                "execution_status": run.status.value,
                "validity_verdict": ValidityVerdict.PROVISIONAL.value,
                "research_progress": "NOT_ASSESSED",
            },
        )
        self._record_idempotency(idempotency_key, req.model_dump(), env)
        return env

    # ── 6. State ──────────────────────────────────────────────────────────────

    def get_state(self, state_id: str, caller: Optional[CallerIdentity] = None) -> ApiResponseEnvelope:
        uow = self.uow_factory()
        with uow:
            state = uow.states.get(state_id)
        if state is None:
            raise EntityNotFoundError("ResearchState", state_id)
        return ApiResponseEnvelope(
            object_id=state.id,
            status="success",
            fingerprint=state.fingerprint(),
            data=state.to_dict(),
        )

    # ── 7. Trajectories ───────────────────────────────────────────────────────

    def get_trajectory(self, run_id: str, caller: Optional[CallerIdentity] = None) -> ApiResponseEnvelope:
        uow = self.uow_factory()
        with uow:
            run = uow.runs.get(run_id)
            outcome = next((o for o in uow.outcomes.list_all() if getattr(o, "run_id", None) == run_id), None)
        if run is None:
            raise EntityNotFoundError("ExperimentRun", run_id)
        return ApiResponseEnvelope(
            object_id=run.id,
            status="success",
            fingerprint=run.fingerprint(),
            data={
                "run": run.to_dict(),
                "outcome": outcome.to_dict() if outcome else None,
                "execution_status": run.status.value,
            },
        )

    # ── 8. Evidence ───────────────────────────────────────────────────────────

    def get_evidence(self, evidence_id: str, caller: Optional[CallerIdentity] = None) -> ApiResponseEnvelope:
        uow = self.uow_factory()
        with uow:
            ev = uow.evidence.get(evidence_id)
        if ev is None:
            raise EntityNotFoundError("Evidence", evidence_id)
        return ApiResponseEnvelope(
            object_id=ev.id,
            status="success",
            fingerprint=ev.fingerprint(),
            data=ev.to_dict(),
        )

    # ── 9. Failures ───────────────────────────────────────────────────────────

    def get_failure(self, failure_id: str, caller: Optional[CallerIdentity] = None) -> ApiResponseEnvelope:
        uow = self.uow_factory()
        with uow:
            fail = uow.failures.get(failure_id)
        if fail is None:
            raise EntityNotFoundError("Failure", failure_id)
        return ApiResponseEnvelope(
            object_id=fail.id,
            status="success",
            fingerprint=fail.fingerprint(),
            data=fail.to_dict(),
        )

    # ── 10. Policy Decisions ──────────────────────────────────────────────────

    def get_policy_decision(self, decision_id: str, caller: Optional[CallerIdentity] = None) -> ApiResponseEnvelope:
        uow = self.uow_factory()
        with uow:
            rec = uow.policy_decisions.get(decision_id)
        if rec is None:
            raise EntityNotFoundError("PolicyDecisionRecord", decision_id)
        return ApiResponseEnvelope(
            object_id=rec.id,
            status="success",
            fingerprint=rec.decision_fingerprint(),
            data=rec.telemetry_dict(),
        )

    # ── 11. Portfolio ─────────────────────────────────────────────────────────

    def get_portfolio(self, branch_id: str, caller: Optional[CallerIdentity] = None) -> ApiResponseEnvelope:
        uow = self.uow_factory()
        with uow:
            branch = uow.portfolio_branches.get(branch_id)
        if branch is None:
            raise EntityNotFoundError("PortfolioBranch", branch_id)
        return ApiResponseEnvelope(
            object_id=branch.id,
            status="success",
            data=branch.to_dict(),
        )

    # ── 12. Saturation ────────────────────────────────────────────────────────

    def get_saturation(self, report_id: str, caller: Optional[CallerIdentity] = None) -> ApiResponseEnvelope:
        uow = self.uow_factory()
        with uow:
            sat = uow.saturation_reports.get(report_id)
        if sat is None:
            raise EntityNotFoundError("SaturationReport", report_id)
        return ApiResponseEnvelope(
            object_id=sat.id,
            status="success",
            data=sat.to_dict(),
        )

    # ── 13. Provenance ────────────────────────────────────────────────────────

    def get_provenance(self, provenance_id: str, caller: Optional[CallerIdentity] = None) -> ApiResponseEnvelope:
        uow = self.uow_factory()
        with uow:
            prov = uow.provenance.get(provenance_id)
        if prov is None:
            raise EntityNotFoundError("Provenance", provenance_id)
        return ApiResponseEnvelope(
            object_id=prov.id,
            status="success",
            fingerprint=prov.fingerprint(),
            data=prov.to_dict(),
        )

    # ── 14. Artifacts ─────────────────────────────────────────────────────────

    def get_artifact(self, artifact_id: str, caller: Optional[CallerIdentity] = None) -> ApiResponseEnvelope:
        uow = self.uow_factory()
        with uow:
            art = uow.artifacts.get(artifact_id)
        if art is None:
            raise EntityNotFoundError("Artifact", artifact_id)
        return ApiResponseEnvelope(
            object_id=art.id,
            status="success",
            fingerprint=art.fingerprint(),
            data=art.to_dict(),
        )

    # ── 15. Canonical Adapter Manifest ────────────────────────────────────────

    def get_adapter_manifest(self, caller: Optional[CallerIdentity] = None) -> ApiResponseEnvelope:
        manifest = self._manifest
        return ApiResponseEnvelope(
            object_id=manifest.id,
            status="success",
            data=manifest.to_dict(),
        )
