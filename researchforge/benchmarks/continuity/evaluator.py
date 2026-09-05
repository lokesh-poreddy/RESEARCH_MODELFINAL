"""researchforge/benchmarks/continuity/evaluator.py — Independent Continuity Evaluator.

Scientific Integrity Invariants (Phase 11):
1. Evaluator operates independently of ResearchPolicy (policy cannot score its own success).
2. Transfer classification requires controlled within-task baseline comparison (experienced vs cold-start).
3. Evaluates decisions and failure avoidance, not merely final model metrics.
4. Preserves full provenance for every audited transfer event.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from ...pipeline.controller import TrialRecord
from .models import TransferClassification, TransferEvent


class ContinuityEvaluator:
    """Computes objective, independent continuity metrics and transfer classifications."""

    def __init__(self, delta_threshold: float = 0.01, dispersion_threshold: float = 0.15) -> None:
        self.delta_threshold = delta_threshold
        self.dispersion_threshold = dispersion_threshold

    def classify_transfer(
        self,
        source_task: str,
        target_task: str,
        strategy: str,
        model_type: str,
        experienced_result: float,
        baseline_without_transfer: float,
        sample_count: int = 1,
        context_level: Optional[int] = None,
        dispersion: float = 0.0,
        provenance_id: str = "",
        rationale: str = "",
        transfer_expected: Optional[bool] = None,
    ) -> TransferEvent:
        """Classifies knowledge transfer by contrasting against controlled within-task baseline."""
        delta = experienced_result - baseline_without_transfer

        # Invariant: If sample count is 0 or query backed off to level 0, evidence is insufficient
        if sample_count == 0 or context_level == 0:
            classification = TransferClassification.INSUFFICIENT_EVIDENCE
            expected = False if transfer_expected is None else transfer_expected
        elif dispersion > self.dispersion_threshold:
            classification = TransferClassification.UNCERTAIN
            expected = False if transfer_expected is None else transfer_expected
        elif delta > self.delta_threshold:
            classification = TransferClassification.POSITIVE
            expected = True if transfer_expected is None else transfer_expected
        elif delta < -self.delta_threshold:
            classification = TransferClassification.NEGATIVE
            expected = False if transfer_expected is None else transfer_expected
        else:
            classification = TransferClassification.NEUTRAL
            expected = False if transfer_expected is None else transfer_expected

        return TransferEvent(
            id=f"xferev_{source_task}_{target_task}_{strategy}_{model_type}",
            schema_version="1.0",
            source_task=source_task,
            target_task=target_task,
            strategy=strategy,
            model_type=model_type,
            baseline_without_transfer=round(baseline_without_transfer, 4),
            experienced_result=round(experienced_result, 4),
            delta=round(delta, 4),
            transfer_expected=expected,
            transfer_classification=classification,
            provenance_id=provenance_id,
            context_level=context_level,
            rationale=rationale,
        )

    def calculate_decision_quality(
        self,
        trials: List[TrialRecord],
        failed_signatures_before: Set[Tuple[str, str]],
    ) -> float:
        """Evaluates fraction of decisions that avoided known failed signatures and produced valid progress."""
        if not trials:
            return 0.0
        good_decisions = 0
        current_fails = set(failed_signatures_before)

        for t in trials:
            sig = (t.strategy, t.model_type)
            # Avoided known prior failure
            avoided_known_failure = sig not in current_fails
            # Did not cause negative transfer
            safe_from_negative_transfer = not t.memory_negative_transfer
            if avoided_known_failure and safe_from_negative_transfer:
                good_decisions += 1

            if t.failure != "None":
                current_fails.add(sig)

        return round(good_decisions / len(trials), 4)

    def calculate_portfolio_diversity(self, trials: List[TrialRecord]) -> float:
        """Computes normalized Shannon entropy of chosen strategies across trials."""
        if not trials:
            return 0.0
        counts = Counter(t.strategy for t in trials)
        n = len(trials)
        k = len(counts)
        if k <= 1:
            return 0.0
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
        max_entropy = math.log2(k)
        return round(min(1.0, max(0.0, entropy / max_entropy)), 4)

    def calculate_policy_stability(self, trials: List[TrialRecord]) -> float:
        """Measures stability of strategy selection across consecutive generations."""
        if len(trials) < 2:
            return 1.0
        switches = sum(1 for i in range(1, len(trials)) if trials[i].strategy != trials[i - 1].strategy)
        # Stability is 1 - switch_rate
        return round(1.0 - (switches / (len(trials) - 1)), 4)

    def calculate_redundant_experiments(self, trials: List[TrialRecord]) -> int:
        """Counts trials that repeated an already failed (strategy, model_type) pair."""
        seen_fails = set()
        redundant = 0
        for t in trials:
            sig = (t.strategy, t.model_type)
            if sig in seen_fails:
                redundant += 1
            if t.failure != "None":
                seen_fails.add(sig)
        return redundant

    def calculate_search_efficiency(self, trials: List[TrialRecord], target_metric: float) -> int:
        """Generation index of first trial that met or exceeded target metric."""
        for t in trials:
            if t.metric >= target_metric:
                return t.generation
        return len(trials)
