"""researchforge/transfer/sampling.py — Phase 12C Active Counterfactual Sampling.

Decides when to spend research budget executing paired counterfactuals (transfer
arm + fresh arm) to observe empirical Delta and update the Utility Predictor.

Sampling Policy:
  Three-tier trigger system:
    1. Hard budget constraint: never exceed budget_fraction * total_gens.
    2. Epistemic uncertainty trigger: sample when pred_var > threshold (active learning).
    3. Decaying epsilon exploration: ensure occasional sampling even when confident.

  Edge cases handled:
    - If total_generations == 0, no sampling is ever triggered (divide-by-zero safe).
    - If budget_fraction == 0.0, always returns False (completely disabled).
    - All counter increments are atomic within should_sample().
"""
from __future__ import annotations

import random
from typing import Any, Dict


class CounterfactualSamplingPolicy:
    """Manages the counterfactual sampling budget B_cf and triggers paired runs."""

    def __init__(
        self,
        budget_fraction: float = 0.15,
        uncertainty_threshold: float = 0.10,
        seed: int = 0,
    ) -> None:
        """
        Args:
            budget_fraction: Maximum fraction of total generations allowed for paired runs.
                             Set to 0.0 to completely disable sampling.
            uncertainty_threshold: Epistemic variance (Var[Delta]) threshold that
                                   triggers a sample. Lowered from 0.15 → 0.10 to
                                   trigger earlier in the posterior's cold-start phase.
            seed: Random seed for the epsilon-exploration decisions.
        """
        if not (0.0 <= budget_fraction <= 1.0):
            raise ValueError(f"budget_fraction must be in [0,1], got {budget_fraction}")

        self.budget_fraction = budget_fraction
        self.uncertainty_threshold = uncertainty_threshold
        self.rng = random.Random(seed)

        self.samples_taken: int = 0
        self.total_decisions: int = 0

        # Track why each sample was triggered for telemetry
        self._trigger_counts: Dict[str, int] = {
            "uncertainty": 0,
            "epsilon": 0,
        }

    def should_sample(
        self,
        pred_var: float,
        generation: int,
        total_generations: int,
    ) -> bool:
        """Determines whether to trigger a paired counterfactual experiment.

        Args:
            pred_var: Epistemic variance of the transfer utility prediction.
            generation: Current (0-indexed) generation.
            total_generations: Total budget for the run (must be > 0).

        Returns:
            True if a paired counterfactual experiment should be executed.
        """
        self.total_decisions += 1

        # Guard: disabled or degenerate run
        if self.budget_fraction == 0.0 or total_generations <= 0:
            return False

        # 1. Hard budget constraint
        max_samples = max(1, int(total_generations * self.budget_fraction))
        if self.samples_taken >= max_samples:
            return False

        # 2. Epistemic Uncertainty Trigger (Active Learning)
        if pred_var > self.uncertainty_threshold:
            self.samples_taken += 1
            self._trigger_counts["uncertainty"] += 1
            return True

        # 3. Decaying epsilon-greedy exploration
        #    Starts at epsilon_0=0.20 and decays linearly to 0.01
        progress = generation / max(1, total_generations - 1)
        epsilon = 0.20 * (1.0 - progress) + 0.01 * progress
        if self.rng.random() < epsilon:
            self.samples_taken += 1
            self._trigger_counts["epsilon"] += 1
            return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Returns sampling telemetry."""
        return {
            "samples_taken": self.samples_taken,
            "total_decisions": self.total_decisions,
            "sampling_rate": self.samples_taken / max(1, self.total_decisions),
            "trigger_counts": dict(self._trigger_counts),
        }
