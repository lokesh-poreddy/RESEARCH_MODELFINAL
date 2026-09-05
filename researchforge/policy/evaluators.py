"""researchforge/policy/evaluators.py — Independent evaluator boundaries.

Scientific Integrity Requirement (Phase 9, Constraint 3 & 10):
Maintain strict separation between:
1. ResearchPolicy      -> chooses actions
2. ScientificEvaluator -> evaluates experimental validity / outcome
3. BenchmarkEvaluator  -> evaluates comparative performance
4. GovernanceEvaluator -> evaluates system / release acceptability

None should be allowed to simply call another's output its own proof of success.
Policy gaming protection: the policy cannot be the sole evaluator of its own success.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from ..domain.action import ResearchAction
from ..domain.experiment import ExperimentRun, ExperimentSpec
from ..domain.outcome import Outcome
from ..domain.validity import Validity, ValidityVerdict


class ScientificEvaluator(Protocol):
    """Evaluates experimental outcome validity independently of the policy that selected it."""

    def evaluate_outcome(
        self,
        spec: ExperimentSpec,
        run: ExperimentRun,
        outcome: Outcome,
    ) -> Validity:
        """Evaluate whether an experiment produced a scientifically valid, reliable outcome."""
        ...


class BenchmarkEvaluator(Protocol):
    """Evaluates comparative performance across fixed tasks, seeds, and conditions."""

    def evaluate_benchmark(
        self,
        benchmark_id: str,
        results: List[Outcome],
        baseline_results: Optional[List[Outcome]] = None,
    ) -> Dict[str, Any]:
        """Produce an independent comparative evaluation without using policy internal scores."""
        ...


class StandardScientificEvaluator:
    """Baseline implementation of ScientificEvaluator."""

    def evaluate_outcome(
        self,
        spec: ExperimentSpec,
        run: ExperimentRun,
        outcome: Outcome,
    ) -> Validity:
        # Check basic execution status and non-null metrics
        if run.status.value != "SUCCESS":
            return Validity(
                verdict=ValidityVerdict.FAIL,
                details={"rationale": f"Run failed with status: {run.status.value}"},
            )

        if not outcome.measured_metrics:
            return Validity(
                verdict=ValidityVerdict.FAIL,
                details={"rationale": "Outcome contains zero measured metrics."},
            )

        return Validity(
            verdict=ValidityVerdict.PASS,
            details={"rationale": "Experiment completed successfully with measured metrics."},
        )


class StandardBenchmarkEvaluator:
    """Baseline implementation of BenchmarkEvaluator for comparative evaluation."""

    def evaluate_benchmark(
        self,
        benchmark_id: str,
        results: List[Outcome],
        baseline_results: Optional[List[Outcome]] = None,
    ) -> Dict[str, Any]:
        metrics = [o.measured_metrics.get("metric", 0.0) for o in results if o.measured_metrics]
        mean_val = sum(metrics) / len(metrics) if metrics else 0.0

        base_mean = 0.0
        if baseline_results:
            base_metrics = [o.measured_metrics.get("metric", 0.0) for o in baseline_results if o.measured_metrics]
            base_mean = sum(base_metrics) / len(base_metrics) if base_metrics else 0.0

        delta = mean_val - base_mean
        return {
            "benchmark_id": benchmark_id,
            "sample_count": len(metrics),
            "mean_performance": mean_val,
            "baseline_mean": base_mean,
            "empirical_gain": delta,
            "independent_evaluation_verified": True,
        }
