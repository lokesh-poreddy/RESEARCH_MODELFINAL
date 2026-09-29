"""researchforge/transfer/sampling.py — Phase 12C Active Counterfactual Sampling.

Decides when to spend research budget executing paired counterfactuals (transfer 
arm + fresh arm) to observe empirical Delta and update the Utility Predictor.
"""
from __future__ import annotations

import random
from typing import Any, Dict


class CounterfactualSamplingPolicy:
    """Manages the counterfactual sampling budget B_cf and triggers paired runs."""
    
    def __init__(self, budget_fraction: float = 0.15, uncertainty_threshold: float = 0.15, seed: int = 0):
        """
        Args:
            budget_fraction: Maximum fraction of total generations allowed for paired runs.
            uncertainty_threshold: Epistemic variance (Var[Delta]) threshold that triggers sampling.
            seed: Random seed for exploration.
        """
        self.budget_fraction = budget_fraction
        self.uncertainty_threshold = uncertainty_threshold
        self.rng = random.Random(seed)
        
        self.samples_taken = 0
        self.total_decisions = 0

    def should_sample(
        self, 
        pred_var: float, 
        generation: int, 
        total_generations: int
    ) -> bool:
        """Determines whether to trigger a paired counterfactual experiment.
        
        Args:
            pred_var: Epistemic variance of the transfer utility prediction.
            generation: Current generation index.
            total_generations: Total budget for the run.
        """
        self.total_decisions += 1
        
        # 1. Hard budget constraint
        max_samples = int(total_generations * self.budget_fraction)
        if self.samples_taken >= max_samples:
            return False
            
        # 2. Epistemic Uncertainty Trigger (Active Learning)
        # If the utility predictor is highly uncertain about this specific context/strategy
        # combination, it is worth spending budget to learn the true Delta.
        if pred_var > self.uncertainty_threshold:
            self.samples_taken += 1
            return True
            
        # 3. Epsilon exploration (decaying random sampling to prevent local minima)
        # Ensures we occasionally sample even when confident, in case the environment shifted.
        epsilon = max(0.01, 0.10 * (1.0 - (generation / max(1, total_generations))))
        if self.rng.random() < epsilon:
            self.samples_taken += 1
            return True
            
        return False
        
    def get_stats(self) -> Dict[str, Any]:
        """Returns sampling telemetry."""
        return {
            "samples_taken": self.samples_taken,
            "total_decisions": self.total_decisions,
            "sampling_rate": self.samples_taken / max(1, self.total_decisions)
        }
