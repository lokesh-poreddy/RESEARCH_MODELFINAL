"""researchforge/benchmarks/cohort/models.py — Benchmark Cohort Specification Models.

Phase 12B.1 Core Invariants:
1. TaskFamily vs. TransferRegime: Transfer regime (SAME_FAMILY, CROSS_FAMILY, UNRELATED)
   is NOT an intrinsic property of a task; it is computed dynamically from the relationship
   between accumulated source experience family and target task family.
2. Canonical Sign Convention: Δ = test_value - control_value, where higher-is-better endpoints
   require Δ > 0 (GREATER) and lower-is-better / degradation endpoints require Δ < 0 (LESS).
3. Explicit Experimental Conditions: All 6 conditions declare exact boolean capability flags.
4. Exact Hypothesis Contrasts: H1-H4 specify fixed test, control, regime, endpoint, and direction.
5. Derived Execution Budget: Total budget is mathematically derived from components rather than
   stored as an independently editable total.
6. Failure & Exclusion Semantics: Distinguishes SCIENTIFIC_FAILURE, EXECUTION_FAILURE,
   INVALID_RUN, and INCONCLUSIVE_RUN, all permanently retained in raw artifacts.
7. Deep Immutability: BenchmarkCohortSpecification raises CohortMutationError on any post-seal modification.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...domain.base import _canonical_json


class CohortMutationError(Exception):
    """Raised when an attempt is made to alter a sealed BenchmarkCohortSpecification."""
    pass


class TaskFamily(str, Enum):
    """Intrinsic domain and structural family of an individual task."""
    DIGITS_SPATIAL = "DIGITS_SPATIAL"      # Grayscale 8x8 spatial pixels, digit recognition
    ECG_TEMPORAL = "ECG_TEMPORAL"          # 1D continuous waveform signals, Gaussian morphology
    XOR_TABULAR = "XOR_TABULAR"            # Discrete tabular non-linear parity logic


class TransferRegime(str, Enum):
    """Transfer relationship between accumulated source experience and target task."""
    NO_PRIOR_EXPERIENCE = "NO_PRIOR_EXPERIENCE"  # Zero accumulated prior experience (fresh baseline)
    SAME_FAMILY = "SAME_FAMILY"                  # Source and target share the identical task family
    CROSS_FAMILY = "CROSS_FAMILY"                # Source and target belong to related continuous/sensory families
    UNRELATED = "UNRELATED"                      # Mismatched domain (e.g. discrete combinatorial vs spatial/signal)


def determine_transfer_regime(
    source_family: Optional[TaskFamily],
    target_family: TaskFamily,
) -> TransferRegime:
    """Deterministically computes the transfer regime from source experience to target task.

    Rules:
    - If source_family is None (first task in sequence, zero prior experience): NO_PRIOR_EXPERIENCE
    - If source_family == target_family: SAME_FAMILY
    - If {source_family, target_family} == {TaskFamily.DIGITS_SPATIAL, TaskFamily.ECG_TEMPORAL}: CROSS_FAMILY
    - If either is XOR_TABULAR and families are not equal: UNRELATED
    """
    if source_family is None:
        return TransferRegime.NO_PRIOR_EXPERIENCE

    if source_family == target_family:
        return TransferRegime.SAME_FAMILY

    pair = {source_family, target_family}
    if pair == {TaskFamily.DIGITS_SPATIAL, TaskFamily.ECG_TEMPORAL}:
        return TransferRegime.CROSS_FAMILY

    return TransferRegime.UNRELATED


def determine_sequence_transfer_regime(
    prior_task_families: Sequence[TaskFamily],
    target_family: TaskFamily,
) -> TransferRegime:
    """Evaluates transfer regime for a sequential history of prior task families into target.

    Rules:
    - Empty prior history -> NO_PRIOR_EXPERIENCE (zero accumulated source tasks)
    - All prior families match target -> SAME_FAMILY
    - All prior families and target are within {DIGITS_SPATIAL, ECG_TEMPORAL} -> CROSS_FAMILY
    - Any involvement of XOR_TABULAR with non-identical family -> UNRELATED
    """
    if not prior_task_families:
        return TransferRegime.NO_PRIOR_EXPERIENCE

    if all(f == target_family for f in prior_task_families):
        return TransferRegime.SAME_FAMILY

    allowed_cross = {TaskFamily.DIGITS_SPATIAL, TaskFamily.ECG_TEMPORAL}
    if all(f in allowed_cross for f in prior_task_families) and target_family in allowed_cross:
        return TransferRegime.CROSS_FAMILY

    return TransferRegime.UNRELATED


class SignConvention(str, Enum):
    """Canonical test-minus-control sign convention identifier."""
    CANONICAL_TEST_MINUS_CONTROL = "CANONICAL_TEST_MINUS_CONTROL"


class Direction(str, Enum):
    """Expected hypothesis effect direction."""
    GREATER = "GREATER"            # Test > Control (Δ > 0)
    LESS = "LESS"                  # Test < Control (Δ < 0)
    NON_MONOTONIC = "NON_MONOTONIC"# Multi-horizon sequence contrast (C1 > 0, C2 < 0)


@dataclass(frozen=True)
class SignConventionSpec:
    """Formal mathematical contract for endpoint delta evaluation."""
    convention_name: SignConvention = SignConvention.CANONICAL_TEST_MINUS_CONTROL
    formula: str = "delta = test_value - control_value"
    higher_is_better_direction: str = "GREATER"
    lower_is_better_direction: str = "LESS"

    @staticmethod
    def compute_delta(test_value: float, control_value: float) -> float:
        """Computes Δ = test_value - control_value."""
        return float(test_value - control_value)

    @staticmethod
    def is_direction_satisfied(delta: float, direction: str, tolerance: float = 1e-9) -> bool:
        """Checks if observed delta satisfies expected direction."""
        if direction == "GREATER":
            return delta > tolerance
        elif direction == "LESS":
            return delta < -tolerance
        raise ValueError(f"Direct sign evaluation not applicable for composite direction: {direction}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "convention_name": self.convention_name.value,
            "formula": self.formula,
            "higher_is_better_direction": self.higher_is_better_direction,
            "lower_is_better_direction": self.lower_is_better_direction,
        }


class BenchmarkCondition(str, Enum):
    """The 6 exact Phase 12B experimental conditions."""
    COLD_START = "COLD_START"
    NO_MEMORY = "NO_MEMORY"
    FLAT_ECRM = "FLAT_ECRM"
    TRAJECTORY_MEMORY = "TRAJECTORY_MEMORY"
    ADAPTIVE_TRAJECTORY = "ADAPTIVE_TRAJECTORY"
    CONTINUOUS_EXPERIENCE = "CONTINUOUS_EXPERIENCE"


@dataclass(frozen=True)
class ConditionCapabilitySpec:
    """Pre-registered capability flags for an experimental condition."""
    condition_id: BenchmarkCondition
    memory_enabled: bool
    trajectory_enabled: bool
    adaptive_retrieval_enabled: bool
    cross_task_experience_allowed: bool
    research_policy_learning_enabled: bool
    prior_task_information_available: bool
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id.value,
            "memory_enabled": self.memory_enabled,
            "trajectory_enabled": self.trajectory_enabled,
            "adaptive_retrieval_enabled": self.adaptive_retrieval_enabled,
            "cross_task_experience_allowed": self.cross_task_experience_allowed,
            "research_policy_learning_enabled": self.research_policy_learning_enabled,
            "prior_task_information_available": self.prior_task_information_available,
            "description": self.description,
        }


class RunOutcomeCategory(str, Enum):
    """Taxonomy of run outcome classifications retained in raw artifacts."""
    VALID_COMPLETED = "VALID_COMPLETED"        # Completed normally, adheres to contract
    SCIENTIFIC_FAILURE = "SCIENTIFIC_FAILURE"  # Completed research cycle but failed target objective
    EXECUTION_FAILURE = "EXECUTION_FAILURE"    # Crashed, timed out, infrastructure failure
    INVALID_RUN = "INVALID_RUN"                # Violated scientific validity or benchmark contract
    INCONCLUSIVE_RUN = "INCONCLUSIVE_RUN"      # Finished but lacks sufficient telemetry for assessment


@dataclass(frozen=True)
class ExclusionPolicySpec:
    """Exclusion rules prohibiting post-hoc filtering while preserving audit integrity."""
    policy_name: str = "canonical_endpoint_specific_exclusion"
    allow_post_hoc_task_removal: bool = False
    allow_post_hoc_seed_removal: bool = False
    allow_post_hoc_ordering_removal: bool = False
    allow_outlier_deletion: bool = False
    allow_bad_result_deletion: bool = False
    allow_manual_removal: bool = False
    rule_statement: str = (
        "Invalid execution artifacts are retained permanently and excluded from the "
        "affected endpoint only when the pre-registered validity rule declares them ineligible."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "allow_post_hoc_task_removal": self.allow_post_hoc_task_removal,
            "allow_post_hoc_seed_removal": self.allow_post_hoc_seed_removal,
            "allow_post_hoc_ordering_removal": self.allow_post_hoc_ordering_removal,
            "allow_outlier_deletion": self.allow_outlier_deletion,
            "allow_bad_result_deletion": self.allow_bad_result_deletion,
            "allow_manual_removal": self.allow_manual_removal,
            "rule_statement": self.rule_statement,
        }


@dataclass(frozen=True)
class TaskGenerationSpec:
    """Deterministic recipe and immutable contract for generating a benchmark task."""
    task_id: str
    task_family: TaskFamily
    generation_algorithm: str
    generator_version: str
    dataset_seed: int
    sample_count_train: int
    sample_count_val: int
    sample_count_test: int
    feature_dimensions: Tuple[int, ...]
    label_construction: str
    noise_parameters: Dict[str, Any]
    train_val_test_splits: Dict[str, Any]
    normalization: str
    representation: str
    target_metric_name: str
    metric_direction: str
    target_metric_threshold: float
    task_fingerprint: str = ""

    def compute_fingerprint(self) -> str:
        payload = {
            "task_id": self.task_id,
            "task_family": self.task_family.value,
            "generation_algorithm": self.generation_algorithm,
            "generator_version": self.generator_version,
            "dataset_seed": self.dataset_seed,
            "sample_count_train": self.sample_count_train,
            "sample_count_val": self.sample_count_val,
            "sample_count_test": self.sample_count_test,
            "feature_dimensions": list(self.feature_dimensions),
            "label_construction": self.label_construction,
            "noise_parameters": self.noise_parameters,
            "train_val_test_splits": self.train_val_test_splits,
            "normalization": self.normalization,
            "representation": self.representation,
            "target_metric_name": self.target_metric_name,
            "metric_direction": self.metric_direction,
            "target_metric_threshold": self.target_metric_threshold,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_family": self.task_family.value,
            "generation_algorithm": self.generation_algorithm,
            "generator_version": self.generator_version,
            "dataset_seed": self.dataset_seed,
            "sample_count_train": self.sample_count_train,
            "sample_count_val": self.sample_count_val,
            "sample_count_test": self.sample_count_test,
            "feature_dimensions": list(self.feature_dimensions),
            "label_construction": self.label_construction,
            "noise_parameters": self.noise_parameters,
            "train_val_test_splits": self.train_val_test_splits,
            "normalization": self.normalization,
            "representation": self.representation,
            "target_metric_name": self.target_metric_name,
            "metric_direction": self.metric_direction,
            "target_metric_threshold": self.target_metric_threshold,
            "task_fingerprint": self.task_fingerprint or self.compute_fingerprint(),
        }


@dataclass(frozen=True)
class CohortHypothesisContrast:
    """Pre-registered hypothesis comparison specification for the cohort."""
    hypothesis_id: str
    title: str
    test_condition: BenchmarkCondition
    control_condition: BenchmarkCondition
    regime: Optional[TransferRegime]
    endpoint: str
    direction: str  # "GREATER", "LESS", "NON_MONOTONIC"
    contrast_formula: str
    horizon_definition: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "test_condition": self.test_condition.value,
            "control_condition": self.control_condition.value,
            "regime": self.regime.value if self.regime else None,
            "endpoint": self.endpoint,
            "direction": self.direction,
            "contrast_formula": self.contrast_formula,
            "horizon_definition": self.horizon_definition,
        }


@dataclass(frozen=True)
class ExecutionBudget:
    """Mathematically derived trial and compute budget."""
    n_tasks: int
    n_seeds: int
    n_orderings: int
    n_conditions: int
    generations_per_task: int
    population_size: int
    timeout_per_trial_seconds: float
    trial_definition: str = "one population-member evaluation (individual candidate genome evaluation)"
    wallclock_semantics: str = "theoretical serialized upper bound assuming worst-case timeout per trial, not a predicted wall-clock runtime"

    @property
    def sequences_per_condition(self) -> int:
        return self.n_seeds * self.n_orderings

    @property
    def task_evaluations_per_condition(self) -> int:
        return self.n_tasks * self.sequences_per_condition

    @property
    def trials_per_task_evaluation(self) -> int:
        return self.generations_per_task * self.population_size

    @property
    def trials_per_condition(self) -> int:
        return self.task_evaluations_per_condition * self.trials_per_task_evaluation

    @property
    def total_benchmark_trials(self) -> int:
        return self.n_conditions * self.trials_per_condition

    @property
    def max_wallclock_seconds_per_condition(self) -> float:
        return float(self.trials_per_condition * self.timeout_per_trial_seconds)

    @property
    def max_total_wallclock_seconds(self) -> float:
        return float(self.total_benchmark_trials * self.timeout_per_trial_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_tasks": self.n_tasks,
            "n_seeds": self.n_seeds,
            "n_orderings": self.n_orderings,
            "n_conditions": self.n_conditions,
            "generations_per_task": self.generations_per_task,
            "population_size": self.population_size,
            "timeout_per_trial_seconds": self.timeout_per_trial_seconds,
            "trial_definition": self.trial_definition,
            "wallclock_semantics": self.wallclock_semantics,
            "sequences_per_condition": self.sequences_per_condition,
            "task_evaluations_per_condition": self.task_evaluations_per_condition,
            "trials_per_task_evaluation": self.trials_per_task_evaluation,
            "trials_per_condition": self.trials_per_condition,
            "total_benchmark_trials": self.total_benchmark_trials,
            "max_wallclock_seconds_per_condition": self.max_wallclock_seconds_per_condition,
            "max_total_wallclock_seconds": self.max_total_wallclock_seconds,
        }


class FrozenDict(dict):
    """Dictionary that forbids any addition, removal, or modification once sealed."""
    def __setitem__(self, key: Any, value: Any) -> None:
        raise CohortMutationError(f"Cohort specification dictionary is sealed! Cannot modify '{key}'.")

    def __delitem__(self, key: Any) -> None:
        raise CohortMutationError(f"Cohort specification dictionary is sealed! Cannot delete '{key}'.")

    def clear(self) -> None:
        raise CohortMutationError("Cohort specification dictionary is sealed!")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise CohortMutationError("Cohort specification dictionary is sealed!")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise CohortMutationError("Cohort specification dictionary is sealed!")


@dataclass
class BenchmarkCohortSpecification:
    """Canonical, pre-registered and cryptographically sealed Phase 12B Benchmark Cohort Specification."""
    specification_version: str = "1.0.1"
    parent_fingerprint: str = "b95ef8fa43a49f7d306e7e42a117b45c51186699467945597d06c2daf7c27707"
    superseded_fingerprints: Tuple[str, ...] = ("b95ef8fa43a49f7d306e7e42a117b45c51186699467945597d06c2daf7c27707",)
    title: str = "Phase 12B Research Continuity Benchmark Analysis Dataset Specification"
    code_revision: str = "HEAD"

    sign_convention: SignConventionSpec = field(default_factory=SignConventionSpec)
    tasks: Dict[str, TaskGenerationSpec] = field(default_factory=dict)
    task_families: Dict[str, TaskFamily] = field(default_factory=dict)
    conditions: Dict[str, ConditionCapabilitySpec] = field(default_factory=dict)
    hypotheses: Dict[str, CohortHypothesisContrast] = field(default_factory=dict)

    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    orderings: Tuple[str, ...] = ("forward", "reverse", "cross_first")
    budget: Optional[ExecutionBudget] = None
    exclusion_policy: ExclusionPolicySpec = field(default_factory=ExclusionPolicySpec)
    failure_categories: Tuple[str, ...] = tuple(c.value for c in RunOutcomeCategory)

    is_sealed: bool = False
    sealed_at: str = ""
    cohort_fingerprint: str = ""

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "is_sealed", False) and name not in ("cohort_fingerprint",):
            raise CohortMutationError(
                f"BenchmarkCohortSpecification is sealed and immutable! Attempted to modify '{name}'."
            )
        super().__setattr__(name, value)

    def compute_fingerprint(self) -> str:
        """Computes deterministic SHA-256 fingerprint over canonical JSON serialization."""
        payload = {
            "specification_version": self.specification_version,
            "parent_fingerprint": self.parent_fingerprint,
            "superseded_fingerprints": list(self.superseded_fingerprints),
            "title": self.title,
            "code_revision": self.code_revision,
            "sign_convention": self.sign_convention.to_dict(),
            "tasks": {k: v.to_dict() for k, v in sorted(self.tasks.items())},
            "task_families": {k: v.value for k, v in sorted(self.task_families.items())},
            "conditions": {k: v.to_dict() for k, v in sorted(self.conditions.items())},
            "hypotheses": {k: v.to_dict() for k, v in sorted(self.hypotheses.items())},
            "seeds": list(self.seeds),
            "orderings": list(self.orderings),
            "budget": self.budget.to_dict() if self.budget else {},
            "exclusion_policy": self.exclusion_policy.to_dict(),
            "failure_categories": list(self.failure_categories),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def seal(self) -> "BenchmarkCohortSpecification":
        """Seals the cohort specification into deep immutability."""
        self.sealed_at = datetime.now(timezone.utc).isoformat()
        self.cohort_fingerprint = self.compute_fingerprint()
        self.tasks = FrozenDict(self.tasks)
        self.task_families = FrozenDict(self.task_families)
        self.conditions = FrozenDict(self.conditions)
        self.hypotheses = FrozenDict(self.hypotheses)
        self.is_sealed = True
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specification_version": self.specification_version,
            "parent_fingerprint": self.parent_fingerprint,
            "superseded_fingerprints": list(self.superseded_fingerprints),
            "title": self.title,
            "code_revision": self.code_revision,
            "sign_convention": self.sign_convention.to_dict(),
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "task_families": {k: v.value for k, v in self.task_families.items()},
            "conditions": {k: v.to_dict() for k, v in self.conditions.items()},
            "hypotheses": {k: v.to_dict() for k, v in self.hypotheses.items()},
            "seeds": list(self.seeds),
            "orderings": list(self.orderings),
            "budget": self.budget.to_dict() if self.budget else {},
            "exclusion_policy": self.exclusion_policy.to_dict(),
            "failure_categories": list(self.failure_categories),
            "is_sealed": self.is_sealed,
            "sealed_at": self.sealed_at,
            "cohort_fingerprint": self.cohort_fingerprint or self.compute_fingerprint(),
        }
