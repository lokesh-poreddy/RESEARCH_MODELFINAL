"""researchforge/api/server.py — Canonical ResearchForge REST API.

Phase 10 Invariants:
1. API -> Application Service -> Domain Contracts -> UnitOfWork.
2. Zero direct database / SQLAlchemy imports in the API layer.
3. Canonical Pydantic DTOs and deterministic error contracts.
4. Separate transport status from domain execution status and validity.
5. Explicit Idempotency-Key support with conflict detection.
6. 15 canonical research endpoints (5 POST + 10 GET) + 3 legacy operational endpoints.
"""
from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..genome.model_genome import ModelGenome
from ..genome.operators import apply_strategy, STRATEGIES
from ..memory.ecrm import ECRM
from ..rdg.graph import ResearchDevelopmentGraph
from ..evaluators.sklearn_evaluator import evaluate_genome
from ..diagnosis.failure_taxonomy import diagnose, FailureCategory
from ..safety.sandbox import SafeRunner, ResourceBudget, SafetyStatus
from ..benchmarks.tasks import digits_task

from .dtos import (
    ApiErrorCode,
    ApiErrorEnvelope,
    ApiResponseEnvelope,
    CreateDecisionRequest,
    CreateExperimentRequest,
    CreateHypothesisRequest,
    CreateProblemRequest,
    CreateQuestionRequest,
)
from .services import (
    ApplicationServiceError,
    CallerIdentity,
    EntityNotFoundError,
    IdempotencyConflictError,
    ResearchApplicationService,
)

MODEL_TYPES = ("MLPClassifier", "RandomForestClassifier", "SVC", "LogisticRegression")

app = FastAPI(
    title="ResearchForge-ECRM API",
    version="1.0.0-alpha.3",
    description="Canonical Research Substrate REST API exposing State, Evidence, "
                "Memory, Policy, Decision, Experiment, Outcome, VRDEG, Provenance, "
                "Artifacts, and Adapter Manifest.",
)

# ── Singleton instances & dependencies ────────────────────────────────────────

_rdg = ResearchDevelopmentGraph()
_ecrm = ECRM()
_genomes: Dict[str, ModelGenome] = {}
_rng = random.Random(0)
_task = digits_task(seed=0)
_safe_runner = SafeRunner(ResourceBudget(
    max_experiments=1000, max_wall_time_s=1800.0, per_experiment_timeout_s=15.0))

_app_service = ResearchApplicationService()


def get_app_service() -> ResearchApplicationService:
    return _app_service


def get_caller_identity(request: Request) -> CallerIdentity:
    """Extension point for caller identity and future authorization context."""
    client_id = request.headers.get("X-Client-ID", "anonymous")
    role = request.headers.get("X-Caller-Role", "researcher")
    scopes_hdr = request.headers.get("X-Caller-Scopes", "read,write")
    scopes = tuple(s.strip() for s in scopes_hdr.split(",") if s.strip())
    return CallerIdentity(client_id=client_id, role=role, scopes=scopes)


# ── Middleware ────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_context_middleware(request: Request, call_next: Any) -> Response:
    """Attaches request_id to request state and response headers."""
    req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ── Deterministic Error Envelopes ─────────────────────────────────────────────

@app.exception_handler(ApplicationServiceError)
async def app_service_error_handler(request: Request, exc: ApplicationServiceError) -> JSONResponse:
    status_code = 400
    if exc.code == ApiErrorCode.NOT_FOUND:
        status_code = 404
    elif exc.code == ApiErrorCode.CONFLICT:
        status_code = 409
    elif exc.code == ApiErrorCode.VALIDATION_ERROR:
        status_code = 422
    elif exc.code in (ApiErrorCode.EXECUTION_FAILURE, ApiErrorCode.INTERNAL_ERROR):
        status_code = 500

    req_id = getattr(request.state, "request_id", "req_unknown")
    envelope = ApiErrorEnvelope(
        error_code=exc.code,
        message=exc.message,
        details=exc.details,
        request_id=req_id,
        schema_version="1.0",
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    req_id = getattr(request.state, "request_id", "req_unknown")
    envelope = ApiErrorEnvelope(
        error_code=ApiErrorCode.VALIDATION_ERROR,
        message="Request payload validation failed.",
        details={"errors": exc.errors()},
        request_id=req_id,
        schema_version="1.0",
    )
    return JSONResponse(status_code=422, content=envelope.model_dump())


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    req_id = getattr(request.state, "request_id", "req_unknown")
    code = ApiErrorCode.INTERNAL_ERROR
    if exc.status_code == 404:
        code = ApiErrorCode.NOT_FOUND
    elif exc.status_code == 409:
        code = ApiErrorCode.CONFLICT
    elif exc.status_code in (400, 422):
        code = ApiErrorCode.VALIDATION_ERROR
    elif exc.status_code in (500, 503):
        code = ApiErrorCode.EXECUTION_FAILURE

    envelope = ApiErrorEnvelope(
        error_code=code,
        message=str(exc.detail) if isinstance(exc.detail, str) else "HTTP Exception",
        details={"detail": exc.detail} if not isinstance(exc.detail, str) else {},
        request_id=req_id,
        schema_version="1.0",
    )
    return JSONResponse(status_code=exc.status_code, content=envelope.model_dump())


# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL RESEARCH ENDPOINTS (15 TOTAL: 5 POST + 10 GET)
# ══════════════════════════════════════════════════════════════════════════════

# ── POST (5 Canonical Research Endpoints) ─────────────────────────────────────

@app.post("/research/problems", response_model=ApiResponseEnvelope)
def create_problem(
    req: CreateProblemRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.create_problem(req, idempotency_key=idempotency_key, caller=caller)


@app.post("/research/questions", response_model=ApiResponseEnvelope)
def create_question(
    req: CreateQuestionRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.create_question(req, idempotency_key=idempotency_key, caller=caller)


@app.post("/research/hypotheses", response_model=ApiResponseEnvelope)
def create_hypothesis(
    req: CreateHypothesisRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.create_hypothesis(req, idempotency_key=idempotency_key, caller=caller)


@app.post("/research/decisions", response_model=ApiResponseEnvelope)
def create_decision(
    req: CreateDecisionRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.create_decision(req, idempotency_key=idempotency_key, caller=caller)


@app.post("/research/experiments", response_model=ApiResponseEnvelope)
def create_experiment_canonical(
    req: CreateExperimentRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.create_experiment(req, idempotency_key=idempotency_key, caller=caller)


# ── GET (10 Canonical Research Endpoints) ──────────────────────────────────────

@app.get("/research/states/{id}", response_model=ApiResponseEnvelope)
def get_state(
    id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.get_state(id, caller=caller)


@app.get("/research/trajectories/{id}", response_model=ApiResponseEnvelope)
def get_trajectory(
    id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.get_trajectory(id, caller=caller)


@app.get("/research/evidence/{id}", response_model=ApiResponseEnvelope)
def get_evidence(
    id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.get_evidence(id, caller=caller)


@app.get("/research/failures/{id}", response_model=ApiResponseEnvelope)
def get_failure(
    id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.get_failure(id, caller=caller)


@app.get("/research/policy/decisions/{id}", response_model=ApiResponseEnvelope)
def get_policy_decision(
    id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.get_policy_decision(id, caller=caller)


@app.get("/research/portfolio/{id}", response_model=ApiResponseEnvelope)
def get_portfolio(
    id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.get_portfolio(id, caller=caller)


@app.get("/research/saturation/{id}", response_model=ApiResponseEnvelope)
def get_saturation(
    id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.get_saturation(id, caller=caller)


@app.get("/research/provenance/{id}", response_model=ApiResponseEnvelope)
def get_provenance(
    id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.get_provenance(id, caller=caller)


@app.get("/research/artifacts/{id}", response_model=ApiResponseEnvelope)
def get_artifact(
    id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.get_artifact(id, caller=caller)


@app.get("/research/adapters/manifest", response_model=ApiResponseEnvelope)
def get_adapters_manifest(
    caller: CallerIdentity = Depends(get_caller_identity),
    service: ResearchApplicationService = Depends(get_app_service),
) -> ApiResponseEnvelope:
    return service.get_adapter_manifest(caller=caller)


# ══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY & OPERATIONAL ENDPOINTS (3 OPERATIONAL / LEGACY)
# ══════════════════════════════════════════════════════════════════════════════

class LegacyExperimentRequest(BaseModel):
    hypothesis_text: str = Field(..., min_length=1, max_length=2000)
    model_type: str = "LogisticRegression"


class MutateRequest(BaseModel):
    model_id: str
    strategy: str


class InsightRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    related_node_ids: Optional[List[str]] = Field(default=None, max_length=50)


@app.post("/experiments")
def create_experiment_legacy(req: LegacyExperimentRequest) -> Dict[str, Any]:
    if req.model_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"model_type must be one of {MODEL_TYPES}")
    hyp = _rdg.add_node("Hypothesis", req.hypothesis_text)
    genome = ModelGenome.default(req.model_type, seed=_rng.randint(0, 10_000))
    _genomes[genome.model_id] = genome
    exp = _rdg.add_node("Experiment", f"pending evaluation of {genome.model_id}",
                         attributes={"genome_id": genome.model_id, "status": "pending"})
    _rdg.add_edge(hyp.id, exp.id, "tested-by")
    return {"hypothesis_id": hyp.id, "experiment_id": exp.id, "model_id": genome.model_id}


@app.get("/experiments/{exp_id}")
def get_experiment_legacy(exp_id: str) -> Dict[str, Any]:
    node = _rdg.nodes.get(exp_id)
    if node is None or node.type != "Experiment":
        raise HTTPException(status_code=404, detail="experiment not found")
    return node.to_dict()


@app.post("/experiments/{exp_id}/run")
def run_experiment(exp_id: str) -> Dict[str, Any]:
    node = _rdg.nodes.get(exp_id)
    if node is None or node.type != "Experiment":
        raise HTTPException(status_code=404, detail="experiment not found")
    genome = _genomes.get(node.attributes.get("genome_id"))
    if genome is None:
        raise HTTPException(status_code=404, detail="genome for this experiment not found")

    violations = genome.safety_check()
    if violations:
        node.attributes["status"] = "rejected"
        raise HTTPException(status_code=422, detail={"safety_violations": violations})

    outcome = _safe_runner.run(evaluate_genome, genome, _task.X_train, _task.y_train,
                                _task.X_val, _task.y_val, _task.metric_fn,
                                target=_task.target_metric)
    if outcome.status != SafetyStatus.OK:
        node.attributes["status"] = "failed"
        node.attributes["safety_status"] = outcome.status.value
        code = 503 if outcome.status == SafetyStatus.BUDGET_EXHAUSTED else 500
        raise HTTPException(status_code=code,
                             detail={"safety_status": outcome.status.value, "error": outcome.error})

    exp_result = outcome.value
    failure = diagnose(exp_result)
    node.attributes["status"] = "completed"

    finding = _rdg.add_node("Finding", f"metric={exp_result.metric:.4f} failure={failure.value}",
                             attributes={"metric": exp_result.metric, "failure": failure.value})
    _rdg.add_edge(exp_id, finding.id, "produces")
    claim_id = None
    if failure == FailureCategory.NONE:
        claim = _rdg.add_node("Claim", f"Genome {genome.model_id} reached {exp_result.metric:.4f}")
        _rdg.add_edge(finding.id, claim.id, "supports")
        claim_id = claim.id

    _ecrm.store(text_summary=f"api_run {genome.model_type} {_task.name}",
                context={"task": _task.name, "genome": genome.to_dict()},
                outcome={"metric": exp_result.metric, "success": failure == FailureCategory.NONE,
                         "failure": failure.value},
                strategy="api_run")

    return {"experiment_id": exp_id, "finding_id": finding.id, "claim_id": claim_id,
            "metric": exp_result.metric, "failure": failure.value,
            "duration_s": round(outcome.duration_s, 4)}


@app.post("/models/mutate")
def mutate_model(req: MutateRequest) -> Dict[str, Any]:
    if req.model_id not in _genomes:
        raise HTTPException(status_code=404, detail="model not found")
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"strategy must be one of {STRATEGIES}")
    parent = _genomes[req.model_id]
    child = apply_strategy(req.strategy, parent, _rng, population=list(_genomes.values()))
    if violations := child.safety_check():
        raise HTTPException(status_code=422, detail={"safety_violations": violations})
    _genomes[child.model_id] = child
    return child.to_dict()


@app.get("/memory/query")
def query_memory(text: str = Query(..., min_length=1, max_length=500),
                  k: int = Query(default=5, ge=1, le=20)) -> Dict[str, Any]:
    recs = _ecrm.query(text, k=k)
    return {"results": [{"id": r.id, "text_summary": r.text_summary, "outcome": r.outcome,
                          "strategy": r.strategy} for r in recs]}


@app.post("/insights")
def create_insight(req: InsightRequest) -> Dict[str, Any]:
    node = _rdg.add_node("Insight", req.text)
    for rid in (req.related_node_ids or []):
        if rid in _rdg.nodes:
            try:
                _rdg.add_edge(rid, node.id, "informs", enforce_types=False)
            except Exception:
                pass
    return node.to_dict()


@app.get("/safety/status")
def safety_status() -> Dict[str, Any]:
    return _safe_runner.status_report()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
