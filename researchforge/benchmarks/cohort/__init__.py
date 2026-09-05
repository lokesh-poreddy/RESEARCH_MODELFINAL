"""researchforge/benchmarks/cohort — Phase 12B Benchmark Cohort & Dataset Specification.

Public exports for canonical cohort specification, task families, transfer regimes,
condition capability declarations, and execution budgets.
"""
from __future__ import annotations

from .models import (
    BenchmarkCohortSpecification,
    BenchmarkCondition,
    CohortHypothesisContrast,
    CohortMutationError,
    ConditionCapabilitySpec,
    Direction,
    ExclusionPolicySpec,
    ExecutionBudget,
    FrozenDict,
    RunOutcomeCategory,
    SignConvention,
    SignConventionSpec,
    TaskFamily,
    TaskGenerationSpec,
    TransferRegime,
    determine_sequence_transfer_regime,
    determine_transfer_regime,
)
from .specification import create_canonical_cohort_spec

__all__ = [
    "BenchmarkCohortSpecification",
    "BenchmarkCondition",
    "CohortHypothesisContrast",
    "CohortMutationError",
    "ConditionCapabilitySpec",
    "Direction",
    "ExclusionPolicySpec",
    "ExecutionBudget",
    "FrozenDict",
    "RunOutcomeCategory",
    "SignConvention",
    "SignConventionSpec",
    "TaskFamily",
    "TaskGenerationSpec",
    "TransferRegime",
    "create_canonical_cohort_spec",
    "determine_sequence_transfer_regime",
    "determine_transfer_regime",
]
