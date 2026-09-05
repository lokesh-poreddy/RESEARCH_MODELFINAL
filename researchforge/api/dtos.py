"""researchforge/api/dtos.py — Canonical API Data Transfer Objects & Error Envelopes.

Phase 10 Invariants:
1. Canonical DTOs strictly decouple external API consumers from internal ORM models.
2. Standard Response Envelope separates API transport status from execution status and validity.
3. Deterministic Error Envelope with controlled error codes.
4. Timezone-aware ISO-8601 UTC timestamping.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiErrorCode(str, Enum):
    """Controlled canonical API error codes."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    IMMUTABILITY_VIOLATION = "IMMUTABILITY_VIOLATION"
    TEMPORAL_VIOLATION = "TEMPORAL_VIOLATION"
    POLICY_REJECTION = "POLICY_REJECTION"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ApiResponseEnvelope(BaseModel, Generic[T]):
    """Standard, schema-governed API response envelope."""
    object_id: Optional[str] = None
    schema_version: str = "1.0"
    status: str = "success"  # Transport / API status: "success" | "error"
    provenance: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fingerprint: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    data: Optional[T] = None


class ApiErrorEnvelope(BaseModel):
    """Standard, deterministic API error contract."""
    error_code: ApiErrorCode
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: str
    schema_version: str = "1.0"
    provenance_id: Optional[str] = None


# ── Request DTOs ─────────────────────────────────────────────────────────────

class CreateProblemRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=256)
    description: str = Field(..., min_length=1, max_length=5000)
    domain: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateQuestionRequest(BaseModel):
    problem_id: str = Field(..., min_length=1)
    question_text: str = Field(..., min_length=1, max_length=2000)
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateHypothesisRequest(BaseModel):
    problem_id: str = Field(..., min_length=1)
    question_id: str = Field(..., min_length=1)
    hypothesis_statement: str = Field(..., min_length=1, max_length=3000)
    predicted_effect: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateDecisionRequest(BaseModel):
    problem_id: str = Field(..., min_length=1)
    hypothesis_id: str = Field(..., min_length=1)
    action_type: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1, max_length=2000)
    target_context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    provenance_id: Optional[str] = None


class CreateExperimentRequest(BaseModel):
    decision_id: str = Field(..., min_length=1)
    hypothesis_id: str = Field(..., min_length=1)
    model_type: str = "MLPClassifier"
    strategy: str = "default"
    timeout_seconds: float = 15.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
