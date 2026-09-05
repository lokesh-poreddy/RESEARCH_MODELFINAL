"""researchforge/benchmarks/continuity/models.py — Research Continuity Benchmark contracts.

Phase 11 Invariants:
1. Strict distinction between COLD_START (fresh state, default machinery) and NO_MEMORY (memory disabled).
2. Transfer classification requires controlled within-task baseline, not delta alone.
3. Cryptographic experience partitioning with anti-future leakage enforcement.
4. Non-causal observational definition of experience gain.
5. Learning-curve telemetry across sequential task counts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ...domain.base import DomainObject, _canonical_json


class TaskRegime(str, Enum):
    """Controlled relationship regimes between sequential research tasks."""
    SAME_FAMILY = "SAME_FAMILY"      # Shared domain, visual/signal structure, hyperparameter space
    CROSS_FAMILY = "CROSS_FAMILY"    # Related problem type with altered morphology or representation
    UNRELATED = "UNRELATED"          # Completely mismatched domain where direct transfer risks negative interference


class ContinuityCondition(str, Enum):
    """Primary experimental control conditions for continuity evaluation."""
    COLD_START = "COLD_START"                        # Fresh state for every task; zero prior task memory
    NO_MEMORY = "NO_MEMORY"                          # Memory contribution explicitly disabled within and across tasks
    FLAT_ECRM = "FLAT_ECRM"                          # Accumulated flat ECRM across prior tasks
    TRAJECTORY_MEMORY = "TRAJECTORY_MEMORY"          # Accumulated trajectory memory
    ADAPTIVE_TRAJECTORY = "ADAPTIVE_TRAJECTORY"      # Accumulated adaptive trajectory memory with backoff
    CONTINUOUS_EXPERIENCE = "CONTINUOUS_EXPERIENCE"  # Full unified accumulated experience inherited sequentially


class TransferClassification(str, Enum):
    """Observational categorization of knowledge transfer from prior tasks."""
    POSITIVE = "POSITIVE"                  # Empirical gain relative to controlled within-task baseline
    NEUTRAL = "NEUTRAL"                    # No significant difference from controlled within-task baseline
    NEGATIVE = "NEGATIVE"                  # Degradation / negative transfer relative to within-task baseline
    UNCERTAIN = "UNCERTAIN"                # Inconclusive / high dispersion across evaluation runs
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # Queries backed off to level 0 or below minimum sample threshold


class FutureInformationLeakageError(Exception):
    """Raised when an execution context attempts to access future task data, evidence, or outcomes."""
    pass


@dataclass(frozen=True)
class ExperiencePartition(DomainObject):
    """Cryptographically sealed partition ensuring strict temporal and task isolation."""
    task_id: str
    task_sequence_index: int
    experience_set_before_task_hash: str
    current_task_info_hash: str
    future_information_hash: str
    allowed_task_ids: List[str]
    sealed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_sealed: bool = True

    def verify_access(self, requested_task_id: str) -> None:
        """Enforces that only experience from prior allowed tasks can be queried."""
        if requested_task_id not in self.allowed_task_ids:
            raise FutureInformationLeakageError(
                f"Future information leakage detected! Task '{self.task_id}' (index {self.task_sequence_index}) "
                f"attempted to query unallowed future task '{requested_task_id}'."
            )

    @classmethod
    def create(
        cls,
        task_id: str,
        task_sequence_index: int,
        allowed_task_ids: List[str],
        prior_memory_fingerprint: str,
        current_task_fingerprint: str,
        future_task_ids: List[str],
    ) -> "ExperiencePartition":
        exp_hash = hashlib.sha256(prior_memory_fingerprint.encode("utf-8")).hexdigest()
        curr_hash = hashlib.sha256(current_task_fingerprint.encode("utf-8")).hexdigest()
        fut_raw = "|".join(sorted(future_task_ids))
        fut_hash = hashlib.sha256(fut_raw.encode("utf-8")).hexdigest()

        return cls(
            id=f"part_{task_id}_{task_sequence_index}",
            schema_version="1.0",
            task_id=task_id,
            task_sequence_index=task_sequence_index,
            experience_set_before_task_hash=exp_hash,
            current_task_info_hash=curr_hash,
            future_information_hash=fut_hash,
            allowed_task_ids=list(allowed_task_ids),
        )


@dataclass(frozen=True)
class TransferEvent(DomainObject):
    """Audit record capturing an instance of knowledge transfer evaluated against within-task baseline."""
    source_task: str
    target_task: str
    strategy: str
    model_type: str
    baseline_without_transfer: float   # Controlled within-task baseline (cold start metric on target task)
    experienced_result: float          # Metric achieved using transferred prior experience
    delta: float                       # experienced_result - baseline_without_transfer
    transfer_expected: bool
    transfer_classification: TransferClassification
    provenance_id: str
    context_level: Optional[int] = None
    rationale: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransferEvent":
        data = dict(d)
        if "transfer_classification" in data and isinstance(data["transfer_classification"], str):
            data["transfer_classification"] = TransferClassification(data["transfer_classification"])
        return cls(**data)


@dataclass
class ContinuityObservation:
    """Raw, unaggregated observation record for one condition × ordering × seed × task evaluation."""
    task_id: str
    task_name: str
    regime: str
    order_id: str
    condition: str
    seed: int
    task_sequence_index: int
    prior_task_count: int
    prior_experience_count: int

    # Core performance & efficiency metrics
    best_metric: float
    baseline_cold_start_metric: float
    experience_gain: float             # Observational: best_metric - baseline_cold_start_metric
    research_efficiency: float         # best_metric / trial count
    search_efficiency: int             # Trial index of first target metric hit (or max trials)

    # Diagnostic decision & failure metrics
    failure_repetition_rate: float     # Rate of repeating known failed strategy/model signatures
    negative_transfer_rate: float      # Fraction of memory-assisted decisions leading to degradation
    memory_utility: float              # Delta against NO_MEMORY control
    decision_quality: float            # Ratio of actions avoiding known failures and non-dominated
    successful_transfer_rate: float    # Fraction of transfer events classified as POSITIVE
    harmful_transfer_rate: float       # Fraction of transfer events classified as NEGATIVE
    information_reuse: int             # Number of times prior memory with context level > 0 was consulted
    redundant_experiments: int         # Count of repeated suboptimal trials
    cost_to_threshold: int             # Total trials to reach target metric
    policy_stability: float            # Ranking stability of policy candidate action scores
    portfolio_diversity: float         # Shannon entropy of exploration branches

    # Memory state boundary fingerprints
    memory_fingerprint_before: str
    memory_fingerprint_after: str
    memory_fingerprint_inherited: str

    # Transfer events audited
    transfer_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "regime": self.regime,
            "order_id": self.order_id,
            "condition": self.condition,
            "seed": self.seed,
            "task_sequence_index": self.task_sequence_index,
            "prior_task_count": self.prior_task_count,
            "prior_experience_count": self.prior_experience_count,
            "best_metric": round(self.best_metric, 4),
            "baseline_cold_start_metric": round(self.baseline_cold_start_metric, 4),
            "experience_gain": round(self.experience_gain, 4),
            "research_efficiency": round(self.research_efficiency, 6),
            "search_efficiency": self.search_efficiency,
            "failure_repetition_rate": round(self.failure_repetition_rate, 4),
            "negative_transfer_rate": round(self.negative_transfer_rate, 4),
            "memory_utility": round(self.memory_utility, 4),
            "decision_quality": round(self.decision_quality, 4),
            "successful_transfer_rate": round(self.successful_transfer_rate, 4),
            "harmful_transfer_rate": round(self.harmful_transfer_rate, 4),
            "information_reuse": self.information_reuse,
            "redundant_experiments": self.redundant_experiments,
            "cost_to_threshold": self.cost_to_threshold,
            "policy_stability": round(self.policy_stability, 4),
            "portfolio_diversity": round(self.portfolio_diversity, 4),
            "memory_fingerprint_before": self.memory_fingerprint_before,
            "memory_fingerprint_after": self.memory_fingerprint_after,
            "memory_fingerprint_inherited": self.memory_fingerprint_inherited,
            "transfer_events": self.transfer_events,
        }


@dataclass
class ContinuityArtifact:
    """Canonical versioned machine-readable artifact capturing full benchmark results."""
    benchmark_version: str = "1.0.0"
    code_revision: str = "HEAD"
    rf_version: str = "1.0.0-alpha.3"
    task_definitions: List[Dict[str, Any]] = field(default_factory=list)
    task_fingerprints: Dict[str, str] = field(default_factory=dict)
    task_orderings: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    budgets: Dict[str, Any] = field(default_factory=dict)
    policy_version: str = "baseline_v1"
    memory_configuration: Dict[str, Any] = field(default_factory=dict)
    experience_partitions: List[Dict[str, Any]] = field(default_factory=list)
    raw_observations: List[Dict[str, Any]] = field(default_factory=list)
    transfer_classifications: List[Dict[str, Any]] = field(default_factory=list)
    learning_curve_telemetry: List[Dict[str, Any]] = field(default_factory=list)
    orchestration_overhead_ms: Dict[str, float] = field(default_factory=dict)
    artifact_fingerprint: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_fingerprint(self) -> str:
        payload = {
            "benchmark_version": self.benchmark_version,
            "rf_version": self.rf_version,
            "task_fingerprints": self.task_fingerprints,
            "task_orderings": sorted(self.task_orderings),
            "conditions": sorted(self.conditions),
            "seeds": sorted(self.seeds),
            "raw_observation_count": len(self.raw_observations),
            "transfer_count": len(self.transfer_classifications),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        fp = self.artifact_fingerprint or self.compute_fingerprint()
        return {
            "benchmark_version": self.benchmark_version,
            "code_revision": self.code_revision,
            "rf_version": self.rf_version,
            "task_definitions": self.task_definitions,
            "task_fingerprints": self.task_fingerprints,
            "task_orderings": self.task_orderings,
            "conditions": self.conditions,
            "seeds": self.seeds,
            "budgets": self.budgets,
            "policy_version": self.policy_version,
            "memory_configuration": self.memory_configuration,
            "experience_partitions": self.experience_partitions,
            "raw_observations": self.raw_observations,
            "transfer_classifications": self.transfer_classifications,
            "learning_curve_telemetry": self.learning_curve_telemetry,
            "orchestration_overhead_ms": self.orchestration_overhead_ms,
            "artifact_fingerprint": fp,
            "created_at": self.created_at,
        }
