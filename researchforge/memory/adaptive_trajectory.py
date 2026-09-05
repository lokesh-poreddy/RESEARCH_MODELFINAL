"""Adaptive Trajectory Memory (ECRM): hierarchical context-sensitive memory
with deterministic backoff and evidence-based statistical confidence.

Scientific Objective (Phase 7):
---------------------------------
Address context fragmentation (NR-001) in sparse trajectory memory:
    Level 3: (strategy, model_type, capacity_bucket) [Fine Context]
       ↓ backoff if sample_count < min_context_samples
    Level 2: (strategy, model_type)                 [Model Family]
       ↓ backoff if sample_count < min_context_samples
    Level 1: (strategy)                             [Strategy]
       ↓ backoff if sample_count < min_context_samples
    Level 0: Global Memory                          [Global Prior]

Architectural & Scientific Principles:
----------------------------------------
1. Context Specificity ≠ Evidence Sufficiency ≠ Statistical Confidence:
   Confidence is derived from available sample volume and dispersion (evidence consistency),
   NOT hard-coded by context level (no arbitrary L3=1.0 -> L0=0.4 scale).
2. Deterministic Gate:
   `min_context_samples` triggers backoff deterministically.
3. Negative Evidence Retention:
   Failures and unsuccessful interventions are never discarded during backoff.
4. Failure Transfer Safety:
   `similar_trajectory_recently_failed` returns structured `FailureCheckResult`.
5. Provenance:
   Every record maintains explicit VRDEG node links (hypothesis_id, spec_id, run_id,
   outcome_id, finding_id, provenance_id).
6. Idempotent Storage:
   Repeated storage of identical semantic trajectories does not inflate memory evidence.
7. Diagnostic Control:
   `oracle_backoff` allows querying a fixed context level directly for scientific attribution.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..genome.model_genome import ModelGenome
from .trajectory import CapacityBucket, Stage, capacity_bucket, generation_stage


@dataclass
class AdaptiveTrajectoryRecord:
    """A single research trajectory record carrying VRDEG provenance and execution metadata."""
    id: str
    generation: int
    stage: Stage
    problem_context: str
    parent_model_type: str
    parent_capacity_bucket: CapacityBucket
    strategy: str
    child_model_type: str
    child_capacity_bucket: CapacityBucket
    metric: float
    success: bool
    failure: str

    # VRDEG Provenance Links
    hypothesis_id: str = ""
    spec_id: str = ""
    run_id: str = ""
    outcome_id: str = ""
    finding_id: str = ""
    provenance_id: str = ""
    created_at: float = field(default_factory=time.time)

    def semantic_key(self) -> Tuple[Any, ...]:
        """Deterministic identity for semantic deduplication and idempotent storage."""
        return (
            self.generation,
            self.stage,
            self.strategy,
            self.parent_model_type,
            self.parent_capacity_bucket,
            self.child_model_type,
            self.child_capacity_bucket,
            round(self.metric, 6),
            self.success,
            self.failure,
            self.hypothesis_id,
            self.run_id or self.spec_id,
        )


@dataclass
class ContextualRetrievalResult:
    """The structured result of hierarchical context retrieval."""
    context_level: int                 # 3 (Fine), 2 (Model Family), 1 (Strategy), 0 (Global)
    requested_context_level: int       # Target specificity (typically 3)
    sample_count: int                  # Number of retrieved records at selected level
    evidence_sufficiency: bool         # Whether sample_count >= min_context_samples
    fallback_reason: Optional[str]     # Diagnostic explanation of backoff (None if satisfied at requested)
    success_rate: float                # Empirical P(success) or fallback prior
    mean_metric: Optional[float]       # Mean metric value across retrieved records
    variance: Optional[float]          # Sample variance of metric
    dispersion: Optional[float]        # Sample standard deviation of metric
    confidence: float                  # Evidence-derived statistical confidence in [0.0, 1.0]
    retrieved_records: List[AdaptiveTrajectoryRecord] = field(default_factory=list)
    provenance_refs: List[str] = field(default_factory=list)


@dataclass
class FailureCheckResult:
    """Structured diagnostics for failure transfer safety."""
    has_failed_majority: bool
    context_level: int
    sample_count: int
    failed_records: List[AdaptiveTrajectoryRecord] = field(default_factory=list)
    failed_record_ids: List[str] = field(default_factory=list)
    reason: str = ""

    def __bool__(self) -> bool:
        """Allow evaluating truthiness directly: `if failure_check: ...`."""
        return self.has_failed_majority


class AdaptiveTrajectoryMemory:
    """Hierarchical, context-sensitive trajectory memory with deterministic backoff.

    Retrieval Levels:
      - Level 3: (strategy, model_type, capacity_bucket)
      - Level 2: (strategy, model_type)
      - Level 1: (strategy)
      - Level 0: Global memory (all records)
    """

    def __init__(self, min_context_samples: int = 3, default_prior: float = 0.6) -> None:
        if min_context_samples < 1:
            raise ValueError(f"min_context_samples must be >= 1, got {min_context_samples}")
        self.min_context_samples = min_context_samples
        self.default_prior = default_prior
        self.records: List[AdaptiveTrajectoryRecord] = []
        self._record_ids: Set[str] = set()
        self._semantic_keys: Set[Tuple[Any, ...]] = set()
        self.backoff_telemetry: List[Dict[str, Any]] = []

    def store(self, record: AdaptiveTrajectoryRecord) -> bool:
        """Deterministically and idempotently store a trajectory record.

        Returns True if newly inserted, False if a semantic duplicate was detected.
        """
        if record.id in self._record_ids:
            return False
        sem_key = record.semantic_key()
        if sem_key in self._semantic_keys:
            return False

        self.records.append(record)
        self._record_ids.add(record.id)
        self._semantic_keys.add(sem_key)
        return True

    def _matches_l3(self, strategy: str, model_type: str,
                    bucket: CapacityBucket) -> List[AdaptiveTrajectoryRecord]:
        return [r for r in self.records
                if r.strategy == strategy
                and r.parent_model_type == model_type
                and r.parent_capacity_bucket == bucket]

    def _matches_l2(self, strategy: str, model_type: str) -> List[AdaptiveTrajectoryRecord]:
        return [r for r in self.records
                if r.strategy == strategy
                and r.parent_model_type == model_type]

    def _matches_l1(self, strategy: str) -> List[AdaptiveTrajectoryRecord]:
        return [r for r in self.records if r.strategy == strategy]

    def _matches_l0(self) -> List[AdaptiveTrajectoryRecord]:
        return list(self.records)

    def _calculate_evidence_statistics(
        self,
        records: List[AdaptiveTrajectoryRecord],
        min_samples: int,
    ) -> Tuple[float, Optional[float], Optional[float], Optional[float], float]:
        """Compute (success_rate, mean_metric, variance, dispersion, confidence).

        Scientific Rule:
        Confidence is derived strictly from evidence characteristics (sample volume and dispersion),
        NEVER hard-coded from context level.
        """
        n = len(records)
        if n == 0:
            return (self.default_prior, None, None, None, 0.0)

        success_rate = sum(1 for r in records if r.success) / n
        metrics = [r.metric for r in records]
        mean_metric = sum(metrics) / n

        if n > 1:
            variance = sum((m - mean_metric) ** 2 for m in metrics) / (n - 1)
            dispersion = math.sqrt(max(0.0, variance))
        else:
            variance = 0.0
            dispersion = 0.0

        # Sample volume saturates toward 1.0 as n reaches 2 * min_samples
        sample_factor = min(1.0, n / max(1, min_samples * 2))
        # Dispersion penalty: higher variance/dispersion degrades statistical confidence
        dispersion_penalty = max(0.1, 1.0 - min(1.0, dispersion))
        confidence = round(sample_factor * dispersion_penalty, 4)

        return (success_rate, mean_metric, variance, dispersion, confidence)

    def _build_result(
        self,
        context_level: int,
        requested_level: int,
        records: List[AdaptiveTrajectoryRecord],
        fallback_reason: Optional[str],
        min_samples: int,
    ) -> ContextualRetrievalResult:
        success_rate, mean_metric, variance, dispersion, confidence = (
            self._calculate_evidence_statistics(records, min_samples)
        )
        provenance_refs = []
        for r in records:
            for ref in (r.finding_id, r.outcome_id, r.run_id, r.spec_id, r.hypothesis_id, r.provenance_id):
                if ref and ref not in provenance_refs:
                    provenance_refs.append(ref)

        sufficiency = len(records) >= min_samples

        result = ContextualRetrievalResult(
            context_level=context_level,
            requested_context_level=requested_level,
            sample_count=len(records),
            evidence_sufficiency=sufficiency,
            fallback_reason=fallback_reason,
            success_rate=success_rate,
            mean_metric=mean_metric,
            variance=variance,
            dispersion=dispersion,
            confidence=confidence,
            retrieved_records=list(records),
            provenance_refs=provenance_refs,
        )

        if fallback_reason is not None:
            self.backoff_telemetry.append({
                "requested_level": requested_level,
                "selected_level": context_level,
                "fallback_reason": fallback_reason,
                "sample_count": len(records),
            })

        return result

    def query_oracle(
        self,
        strategy: str,
        model_type: str,
        capacity_bucket: CapacityBucket,
        oracle_level: int,
        min_samples: Optional[int] = None,
    ) -> ContextualRetrievalResult:
        """Diagnostic Oracle Retrieval Mode.

        Directly queries the specified context level without automatic fallback.
        Used as a diagnostic control to distinguish:
          A. Hierarchy design failure
          B. Context selection / backoff failure
          C. Insufficient memory failure
        """
        min_s = min_samples if min_samples is not None else self.min_context_samples
        if oracle_level == 3:
            records = self._matches_l3(strategy, model_type, capacity_bucket)
        elif oracle_level == 2:
            records = self._matches_l2(strategy, model_type)
        elif oracle_level == 1:
            records = self._matches_l1(strategy)
        elif oracle_level == 0:
            records = self._matches_l0()
        else:
            raise ValueError(f"Invalid oracle_level: {oracle_level}. Must be 0, 1, 2, or 3.")

        return self._build_result(
            context_level=oracle_level,
            requested_level=oracle_level,
            records=records,
            fallback_reason=f"oracle_level_forced: level={oracle_level}",
            min_samples=min_s,
        )

    def query_context(
        self,
        strategy: str,
        model_type: str,
        capacity_bucket: CapacityBucket,
        min_samples: Optional[int] = None,
    ) -> ContextualRetrievalResult:
        """Deterministic hierarchical context retrieval with backoff: L3 -> L2 -> L1 -> L0."""
        min_s = min_samples if min_samples is not None else self.min_context_samples

        # Level 3: (strategy, model_type, capacity_bucket)
        l3 = self._matches_l3(strategy, model_type, capacity_bucket)
        if len(l3) >= min_s:
            return self._build_result(3, 3, l3, fallback_reason=None, min_samples=min_s)

        # Level 2: (strategy, model_type)
        l2 = self._matches_l2(strategy, model_type)
        if len(l2) >= min_s:
            reason = f"L3_insufficient_samples: count={len(l3)} < min_samples={min_s}"
            return self._build_result(2, 3, l2, fallback_reason=reason, min_samples=min_s)

        # Level 1: (strategy)
        l1 = self._matches_l1(strategy)
        if len(l1) >= min_s:
            reason = f"L2_insufficient_samples: count={len(l2)} < min_samples={min_s}"
            return self._build_result(1, 3, l1, fallback_reason=reason, min_samples=min_s)

        # Level 0: Global memory
        l0 = self._matches_l0()
        if len(l0) > 0:
            reason = f"L1_insufficient_samples: count={len(l1)} < min_samples={min_s}"
            return self._build_result(0, 3, l0, fallback_reason=reason, min_samples=min_s)

        # Empty memory
        return self._build_result(0, 3, [], fallback_reason="empty_memory", min_samples=min_s)

    def contextual_success_rate(
        self,
        strategy: str,
        model_type: str,
        capacity_bucket: CapacityBucket,
        min_samples: Optional[int] = None,
        oracle_level: Optional[int] = None,
    ) -> ContextualRetrievalResult:
        """Query context-sensitive success rate with full retrieval metadata."""
        if oracle_level is not None:
            return self.query_oracle(strategy, model_type, capacity_bucket, oracle_level, min_samples)
        return self.query_context(strategy, model_type, capacity_bucket, min_samples)

    def similar_trajectory_recently_failed(
        self,
        strategy: str,
        model_type: str,
        capacity_bucket: CapacityBucket,
        window: int = 3,
        min_samples: Optional[int] = None,
        oracle_level: Optional[int] = None,
    ) -> FailureCheckResult:
        """Check whether a majority of recent context-matched trajectories failed.

        Negative Evidence Safety:
        Returns structured FailureCheckResult with diagnostics, failed records, and context level used.
        """
        retrieval = self.contextual_success_rate(
            strategy, model_type, capacity_bucket, min_samples=min_samples, oracle_level=oracle_level
        )
        records = retrieval.retrieved_records
        if not records:
            return FailureCheckResult(
                has_failed_majority=False,
                context_level=retrieval.context_level,
                sample_count=0,
                failed_records=[],
                failed_record_ids=[],
                reason="no_records_in_context",
            )

        recent = records[-window:]
        failed = [r for r in recent if not r.success]
        has_failed = len(failed) > (len(recent) / 2)

        reason = (
            f"majority_failure: {len(failed)}/{len(recent)} failed in window={window} "
            f"at level={retrieval.context_level}"
            if has_failed else
            f"no_majority_failure: {len(failed)}/{len(recent)} failed in window={window} "
            f"at level={retrieval.context_level}"
        )

        return FailureCheckResult(
            has_failed_majority=has_failed,
            context_level=retrieval.context_level,
            sample_count=len(recent),
            failed_records=failed,
            failed_record_ids=[r.id for r in failed],
            reason=reason,
        )

    def evidence_chain_for(self, trajectory_id: str, graph: Any) -> List[Any]:
        """Reconstruct the VRDEG or RDG evidence chain for a given trajectory."""
        rec = next((r for r in self.records if r.id == trajectory_id), None)
        if rec is None:
            return []
        target_id = rec.finding_id or rec.outcome_id or rec.run_id or rec.hypothesis_id
        if not target_id:
            return []
        if hasattr(graph, "evidence_chain"):
            return graph.evidence_chain(target_id)
        if hasattr(graph, "get_experiment_lineage"):
            return graph.get_experiment_lineage(target_id)
        return []

    def stats(self) -> Dict[str, Any]:
        """Return memory composition and backoff statistics."""
        l3_contexts = {(r.strategy, r.parent_model_type, r.parent_capacity_bucket) for r in self.records}
        l2_contexts = {(r.strategy, r.parent_model_type) for r in self.records}
        l1_strategies = {r.strategy for r in self.records}

        level_counts = {3: 0, 2: 0, 1: 0, 0: 0}
        for event in self.backoff_telemetry:
            lvl = event.get("selected_level", 0)
            level_counts[lvl] = level_counts.get(lvl, 0) + 1

        return {
            "total_trajectories": len(self.records),
            "distinct_l3_contexts": len(l3_contexts),
            "distinct_l2_contexts": len(l2_contexts),
            "distinct_l1_strategies": len(l1_strategies),
            "backoff_events_count": len(self.backoff_telemetry),
            "backoff_level_distribution": level_counts,
        }


def new_adaptive_trajectory_id() -> str:
    return f"atraj_{uuid.uuid4().hex[:10]}"
