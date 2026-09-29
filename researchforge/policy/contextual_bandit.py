"""researchforge/policy/contextual_bandit.py — LinUCB contextual policy.

RF-2.0 Phase 4: Replaces the context-free UCB policy from policy_learner.py
with a Linear Upper Confidence Bound (LinUCB) contextual bandit that conditions
action selection on the current research state context.

Scientific Rationale:
  The RF-1 policy_learner.py implements a multi-armed bandit (MAB) with a
  shared Q-value per action. This is provably suboptimal when the optimal
  action depends on context — e.g., "increase_capacity" is good when the
  current model is small (underfitting), but bad when it's already large
  (overfitting risk). LinUCB addresses this by learning a separate linear
  model per action that maps context features → expected reward:

    r̂(a|x) = θ_a^T · x

  with upper confidence bound:

    UCB(a|x) = θ_a^T · x + α · √(x^T · A_a^{-1} · x)

  where A_a is the design matrix for action a, and α controls exploration.

  This is the standard LinUCB from Li et al. (2010), "A Contextual-Bandit
  Approach to Personalized News Article Recommendation", adapted for
  research strategy selection.

Context Features (extracted from ResearchState + PosteriorMemory):
  - Normalized best_metric (performance level)
  - Budget fraction remaining
  - Population diversity (n distinct model types)
  - Memory evidence count at finest available level
  - Posterior success probability for this action
  - Posterior uncertainty for this action
  - Recent improvement trend (last 3 generations)
  - Failure frequency in recent window

PRESERVED: PolicyLearner (UCB-only) remains unchanged for RF-0.x regression
baseline compatibility and the "no_memory" / "random" conditions.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# Context feature dimension (fixed for schema stability)
CONTEXT_DIM = 8


@dataclass
class ContextVector:
    """Structured context representation for the research loop.

    Each field maps to a specific dimension in the LinUCB feature vector.
    All values are normalized to [0, 1] or small bounded ranges.
    """
    best_metric: float = 0.0           # Normalized current best metric [0, 1]
    budget_fraction: float = 1.0       # Fraction of budget remaining [0, 1]
    population_diversity: float = 0.5  # Fraction of distinct model types in population [0, 1]
    memory_evidence: float = 0.0       # Normalized evidence count (min(n/20, 1)) [0, 1]
    posterior_success: float = 0.5     # P(success|action, context) from posterior [0, 1]
    posterior_uncertainty: float = 1.0 # Posterior uncertainty [0, 1]
    improvement_trend: float = 0.0     # Mean metric delta over recent window [-1, 1]
    failure_rate: float = 0.0          # Fraction of recent trials that failed [0, 1]

    def to_array(self) -> np.ndarray:
        """Convert to numpy feature vector."""
        return np.array([
            self.best_metric,
            self.budget_fraction,
            self.population_diversity,
            self.memory_evidence,
            self.posterior_success,
            self.posterior_uncertainty,
            self.improvement_trend,
            self.failure_rate,
        ], dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        return {
            "best_metric": round(self.best_metric, 4),
            "budget_fraction": round(self.budget_fraction, 4),
            "population_diversity": round(self.population_diversity, 4),
            "memory_evidence": round(self.memory_evidence, 4),
            "posterior_success": round(self.posterior_success, 4),
            "posterior_uncertainty": round(self.posterior_uncertainty, 4),
            "improvement_trend": round(self.improvement_trend, 4),
            "failure_rate": round(self.failure_rate, 4),
        }


@dataclass
class ActionScore:
    """Detailed score decomposition for a single action."""
    action: str
    predicted_reward: float   # θ^T x
    exploration_bonus: float  # α √(x^T A^{-1} x)
    ucb_score: float          # predicted_reward + exploration_bonus
    memory_modifier: float    # Transfer gate multiplier [0, 1]
    final_score: float        # ucb_score * memory_modifier
    context: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "predicted_reward": round(self.predicted_reward, 6),
            "exploration_bonus": round(self.exploration_bonus, 6),
            "ucb_score": round(self.ucb_score, 6),
            "memory_modifier": round(self.memory_modifier, 6),
            "final_score": round(self.final_score, 6),
        }


@dataclass
class PolicyDecision:
    """Complete traceable record of a policy decision."""
    chosen_action: str
    action_scores: Dict[str, ActionScore]
    context_used: ContextVector
    exploration_alpha: float
    decision_type: str = "contextual_bandit"  # For provenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen_action": self.chosen_action,
            "decision_type": self.decision_type,
            "exploration_alpha": self.exploration_alpha,
            "context": self.context_used.to_dict(),
            "scores": {k: v.to_dict() for k, v in self.action_scores.items()},
        }


class ContextualBanditPolicy:
    """LinUCB contextual bandit policy for research strategy selection.

    Maintains per-action linear models:
      A_a ∈ R^{d×d}  — design matrix (initialized to I_d for regularization)
      b_a ∈ R^d      — reward-weighted feature accumulator
      θ_a = A_a^{-1} b_a — parameter estimate

    Parameters
    ----------
    actions : List[str]
        Available strategy names.
    alpha : float
        Exploration parameter controlling UCB width. Default 1.0.
        Higher = more exploration, lower = more exploitation.
    context_dim : int
        Dimensionality of context features. Default CONTEXT_DIM.
    rng : random.Random | None
        Random number generator for tie-breaking.
    """

    def __init__(self, actions: List[str], alpha: float = 1.0,
                 context_dim: int = CONTEXT_DIM,
                 rng: Optional[random.Random] = None) -> None:
        self.actions = list(actions)
        self.alpha = alpha
        self.d = context_dim
        self.rng = rng or random.Random()

        # Per-action LinUCB state
        self._A: Dict[str, np.ndarray] = {}
        self._b: Dict[str, np.ndarray] = {}
        self._n_updates: Dict[str, int] = {}
        self.total_trials = 0

        for a in self.actions:
            self._A[a] = np.eye(self.d)
            self._b[a] = np.zeros(self.d)
            self._n_updates[a] = 0

    def add_action(self, action: str) -> None:
        """Dynamically adds a newly discovered action to the bandit."""
        if action not in self.actions:
            self.actions.append(action)
            self._A[action] = np.eye(self.d)
            self._b[action] = np.zeros(self.d)
            self._n_updates[action] = 0

    def select_action(
        self,
        context: ContextVector,
        memory_modifier: Optional[Callable[[str], float]] = None,
        failure_check: Optional[Callable[[str], bool]] = None,
    ) -> PolicyDecision:
        """Select an action using LinUCB with optional memory/failure modifiers.

        Args:
            context: Current research state context vector.
            memory_modifier: Optional function returning a [0,1] multiplier
                per action from the transfer gate / posterior memory.
            failure_check: Optional function returning True if ECRM has seen
                a similar failure for this action (backward compat with RF-1).

        Returns:
            PolicyDecision with full score decomposition for provenance.
        """
        x = context.to_array()
        scores: Dict[str, ActionScore] = {}
        best_action = None
        best_score = -math.inf

        for a in self.actions:
            A_inv = np.linalg.inv(self._A[a])
            theta = A_inv @ self._b[a]

            # Predicted reward
            pred = float(theta @ x)

            # Exploration bonus: α √(x^T A^{-1} x)
            exploration = self.alpha * math.sqrt(float(x @ A_inv @ x))

            ucb = pred + exploration

            # Apply memory modifier
            mm = 1.0
            if memory_modifier is not None:
                mm = memory_modifier(a)
            if failure_check is not None and failure_check(a):
                mm *= 0.5

            final = ucb * mm

            scores[a] = ActionScore(
                action=a,
                predicted_reward=pred,
                exploration_bonus=exploration,
                ucb_score=ucb,
                memory_modifier=mm,
                final_score=final,
            )

            if final > best_score:
                best_score = final
                best_action = a

        return PolicyDecision(
            chosen_action=best_action,
            action_scores=scores,
            context_used=context,
            exploration_alpha=self.alpha,
        )

    def update(self, action: str, context: ContextVector, reward: float) -> None:
        """Update the LinUCB model for the taken action with the observed reward.

        Standard LinUCB update:
          A_a ← A_a + x x^T
          b_a ← b_a + r x
        """
        if action not in self._A:
            raise ValueError(f"Unknown action: {action}")

        x = context.to_array()
        self._A[action] += np.outer(x, x)
        self._b[action] += reward * x
        self._n_updates[action] += 1
        self.total_trials += 1

    def select_random(self) -> str:
        """Uniform random selection (for random baseline condition)."""
        return self.rng.choice(self.actions)

    @property
    def times_tried(self) -> Dict[str, int]:
        """Backward-compatible API for trial counts."""
        return dict(self._n_updates)

    def action_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return per-action model summary for diagnostics."""
        result = {}
        for a in self.actions:
            A_inv = np.linalg.inv(self._A[a])
            theta = A_inv @ self._b[a]
            result[a] = {
                "theta_norm": round(float(np.linalg.norm(theta)), 4),
                "n_updates": self._n_updates[a],
                "theta": [round(float(t), 4) for t in theta],
            }
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy state for persistence/provenance."""
        return {
            "policy_type": "LinUCB",
            "alpha": self.alpha,
            "context_dim": self.d,
            "total_trials": self.total_trials,
            "actions": list(self.actions),
            "n_updates": dict(self._n_updates),
        }
