"""researchforge/api/__init__.py — Canonical API module.

Exposes the FastAPI application, canonical DTOs, application services, and error models.
"""
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
from .server import app, get_app_service

__all__ = [
    "app",
    "get_app_service",
    "ApiErrorCode",
    "ApiErrorEnvelope",
    "ApiResponseEnvelope",
    "CreateDecisionRequest",
    "CreateExperimentRequest",
    "CreateHypothesisRequest",
    "CreateProblemRequest",
    "CreateQuestionRequest",
    "ApplicationServiceError",
    "CallerIdentity",
    "EntityNotFoundError",
    "IdempotencyConflictError",
    "ResearchApplicationService",
]
