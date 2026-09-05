"""researchforge/benchmarks/statistical — Statistical Analysis Plan and Evaluation Engine.

Phase 12A: Pre-registered Statistical Protocol & Downstream Inference Engine.
"""
from .engine import StatisticalAnalysisEngine
from .models import (
    EndpointType,
    HypothesisDecision,
    HypothesisEvaluationResult,
    HypothesisType,
    PairedContrastRecord,
    PairedTestResult,
    SensitivityResult,
    StatisticalArtifact,
    ValidationStatus,
)
from .plan import (
    HypothesisSpec,
    PlanMutationError,
    StatisticalAnalysisPlan,
    create_canonical_sap,
)

__all__ = [
    "EndpointType",
    "HypothesisDecision",
    "HypothesisEvaluationResult",
    "HypothesisType",
    "PairedContrastRecord",
    "PairedTestResult",
    "SensitivityResult",
    "StatisticalArtifact",
    "ValidationStatus",
    "HypothesisSpec",
    "PlanMutationError",
    "StatisticalAnalysisPlan",
    "create_canonical_sap",
    "StatisticalAnalysisEngine",
]
