"""researchforge/memory/posterior.py — Beta-Binomial posterior memory model.

RF-2.0 Phase 3: Replaces the heuristic RES(h,G) = α·Rel + β/(1+NTR) - γ·uncertainty
with a principled Bayesian posterior that provides:
  - Calibrated success probability estimates
  - Evidence-derived uncertainty (not arbitrary mean/std ratios)
  - Natural handling of sparse evidence via informative priors
  - Proper conjugate updating

Scientific Rationale:
  For bounded outcomes (success/failure), the Beta-Binomial model is the
  natural conjugate prior. Instead of computing reliability as μ/σ (which
  can blow up to 87 when σ→0), we compute:

    p(success|a,c) ~ Beta(α + s, β + f)

  where s = successes, f = failures, and (α₀, β₀) = prior.
  The posterior mean E[p] = (α+s)/(α+β+s+f) is always in [0,1].
  The posterior variance Var[p] = (α+s)(β+f)/((α+β+s+f)²(α+β+s+f+1))
  provides calibrated uncertainty.

  This directly addresses the paper's μ/σ scale problem identified in the
  mathematical review.

PRESERVED: This module does NOT replace ECRM. It provides a parallel posterior
model that can be used by the new contextual policy. The RF-0.x RES algorithm
in ecrm.py remains unchanged for regression baseline compatibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PosteriorState:
    """Beta posterior state for a (context, action) pair.

    Maintains conjugate Beta(alpha, beta) parameters updated with observed
    successes and failures. Also tracks continuous metric improvement deltas
    for regression-style outcome modeling.
    """
    alpha: float  # prior + observed successes
    beta: float   # prior + observed failures
    n_observations: int = 0
    sum_delta: float = 0.0        # sum of metric deltas (child - parent)
    sum_delta_sq: float = 0.0     # sum of squared deltas
    last_updated_gen: int = -1

    @property
    def mean(self) -> float:
        """Posterior mean: E[p] = α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Posterior variance: Var[p] = αβ / ((α+β)²(α+β+1))."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def uncertainty(self) -> float:
        """Normalized uncertainty in [0, 1]. Maximum at α=β (uniform),
        minimum as evidence accumulates asymmetrically."""
        # Width of 95% credible interval relative to [0,1]
        # Approximation: 2 * 1.96 * std (Normal approx to Beta)
        return min(1.0, 2.0 * 1.96 * self.std)

    @property
    def confidence(self) -> float:
        """Confidence = 1 - uncertainty. Higher with more evidence."""
        return 1.0 - self.uncertainty

    @property
    def mean_delta(self) -> float:
        """Mean metric improvement when this action was taken."""
        if self.n_observations == 0:
            return 0.0
        return self.sum_delta / self.n_observations

    @property
    def delta_variance(self) -> float:
        """Variance of metric improvement deltas."""
        if self.n_observations < 2:
            return 0.0
        mean = self.mean_delta
        return (self.sum_delta_sq / self.n_observations) - mean * mean

    def update(self, success: bool, metric_delta: float = 0.0,
               generation: int = -1) -> None:
        """Bayesian conjugate update with observed outcome.

        Args:
            success: Whether the action produced a successful outcome
                     (metric >= parent or no failure).
            metric_delta: Continuous improvement signal (child_metric - parent_metric).
            generation: Current generation number for recency tracking.
        """
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.n_observations += 1
        self.sum_delta += metric_delta
        self.sum_delta_sq += metric_delta * metric_delta
        self.last_updated_gen = generation

    def sample(self, rng: Any = None) -> float:
        """Thompson sampling: draw from the posterior Beta distribution.

        Useful for exploration: actions with uncertain posteriors will have
        high-variance draws, naturally encouraging exploration.
        """
        import random as _random
        r = rng if rng is not None else _random
        # Python's random.betavariate requires alpha,beta > 0
        return r.betavariate(max(self.alpha, 1e-6), max(self.beta, 1e-6))

    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Approximate credible interval using Normal approximation to Beta.

        For a more accurate interval with small samples, use scipy.stats.beta.ppf.
        This avoids adding scipy as a hard runtime dependency.
        """
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(level, 1.96)
        lo = max(0.0, self.mean - z * self.std)
        hi = min(1.0, self.mean + z * self.std)
        return (lo, hi)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "n_observations": self.n_observations,
            "sum_delta": self.sum_delta,
            "sum_delta_sq": self.sum_delta_sq,
            "last_updated_gen": self.last_updated_gen,
            "mean": round(self.mean, 6),
            "uncertainty": round(self.uncertainty, 6),
            "confidence": round(self.confidence, 6),
            "mean_delta": round(self.mean_delta, 6),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PosteriorState":
        return cls(
            alpha=float(d["alpha"]),
            beta=float(d["beta"]),
            n_observations=int(d.get("n_observations", 0)),
            sum_delta=float(d.get("sum_delta", 0.0)),
            sum_delta_sq=float(d.get("sum_delta_sq", 0.0)),
            last_updated_gen=int(d.get("last_updated_gen", -1)),
        )


class PosteriorMemory:
    """Hierarchical Beta-Binomial posterior memory with evidence-aware backoff.

    Maintains separate posteriors at multiple context granularities:
      Level 3: (strategy, model_type, capacity_bucket)
      Level 2: (strategy, model_type)
      Level 1: (strategy,)
      Level 0: global

    When queried, uses the most specific level with sufficient evidence,
    falling back to coarser levels using Bayesian hierarchical smoothing:

        p_effective = (n / (n + κ)) · p_specific + (κ / (n + κ)) · p_parent

    This directly addresses NR-001 (context fragmentation) with a principled
    mathematical solution rather than hard backoff thresholds.

    Parameters
    ----------
    prior_alpha : float
        Prior pseudo-count for successes. Default 1.0 (uniform prior).
    prior_beta : float
        Prior pseudo-count for failures. Default 1.0 (uniform prior).
    kappa : float
        Smoothing strength for hierarchical backoff. Higher values give more
        weight to coarser (parent) levels. Default 3.0.
    min_evidence : int
        Minimum observations at a context level before it contributes
        meaningfully to the posterior. Below this, the level defers entirely
        to its parent. Default 2.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0,
                 kappa: float = 3.0, min_evidence: int = 2) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be positive")
        if kappa < 0:
            raise ValueError("kappa must be non-negative")
        if min_evidence < 1:
            raise ValueError("min_evidence must be >= 1")

        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.kappa = kappa
        self.min_evidence = min_evidence

        # Posterior stores keyed by context tuple
        self._level3: Dict[Tuple[str, str, str], PosteriorState] = {}  # (strategy, model_type, bucket)
        self._level2: Dict[Tuple[str, str], PosteriorState] = {}       # (strategy, model_type)
        self._level1: Dict[Tuple[str,], PosteriorState] = {}           # (strategy,)
        self._global: PosteriorState = PosteriorState(
            alpha=prior_alpha, beta=prior_beta
        )

    def _get_or_create(self, store: dict, key: tuple) -> PosteriorState:
        if key not in store:
            store[key] = PosteriorState(
                alpha=self.prior_alpha, beta=self.prior_beta
            )
        return store[key]

    def update(self, strategy: str, model_type: str, capacity_bucket: str,
               success: bool, metric_delta: float = 0.0,
               generation: int = -1) -> None:
        """Update posteriors at ALL context levels with the observed outcome.

        This is correct because each level accumulates its own evidence.
        The hierarchical smoothing happens at query time, not update time.
        """
        # Level 3
        key3 = (strategy, model_type, capacity_bucket)
        self._get_or_create(self._level3, key3).update(success, metric_delta, generation)

        # Level 2
        key2 = (strategy, model_type)
        self._get_or_create(self._level2, key2).update(success, metric_delta, generation)

        # Level 1
        key1 = (strategy,)
        self._get_or_create(self._level1, key1).update(success, metric_delta, generation)

        # Level 0
        self._global.update(success, metric_delta, generation)

    def query(self, strategy: str, model_type: str,
              capacity_bucket: str) -> "PosteriorQueryResult":
        """Query the hierarchical posterior with Bayesian smoothing.

        Returns the effective posterior mean, uncertainty, and the context
        level that was primarily used (for diagnostic/provenance purposes).
        """
        # Gather evidence at each level
        key3 = (strategy, model_type, capacity_bucket)
        key2 = (strategy, model_type)
        key1 = (strategy,)

        p3 = self._level3.get(key3)
        p2 = self._level2.get(key2)
        p1 = self._level1.get(key1)
        p0 = self._global

        # Compute smoothed estimate bottom-up
        # Level 0 (global) — always available
        est_0 = p0.mean
        unc_0 = p0.uncertainty
        n_0 = p0.n_observations

        # Level 1: smooth with global
        if p1 is not None and p1.n_observations >= self.min_evidence:
            n1 = p1.n_observations
            w1 = n1 / (n1 + self.kappa)
            est_1 = w1 * p1.mean + (1.0 - w1) * est_0
            unc_1 = w1 * p1.uncertainty + (1.0 - w1) * unc_0
        else:
            est_1 = est_0
            unc_1 = unc_0
            n1 = 0

        # Level 2: smooth with level 1
        if p2 is not None and p2.n_observations >= self.min_evidence:
            n2 = p2.n_observations
            w2 = n2 / (n2 + self.kappa)
            est_2 = w2 * p2.mean + (1.0 - w2) * est_1
            unc_2 = w2 * p2.uncertainty + (1.0 - w2) * unc_1
        else:
            est_2 = est_1
            unc_2 = unc_1
            n2 = 0

        # Level 3: smooth with level 2
        if p3 is not None and p3.n_observations >= self.min_evidence:
            n3 = p3.n_observations
            w3 = n3 / (n3 + self.kappa)
            est_3 = w3 * p3.mean + (1.0 - w3) * est_2
            unc_3 = w3 * p3.uncertainty + (1.0 - w3) * unc_2
            primary_level = 3
        elif n2 > 0:
            est_3 = est_2
            unc_3 = unc_2
            primary_level = 2
        elif n1 > 0:
            est_3 = est_1
            unc_3 = unc_1
            primary_level = 1
        else:
            est_3 = est_0
            unc_3 = unc_0
            primary_level = 0

        # Confidence derived from evidence quantity AND posterior concentration
        confidence = max(0.0, 1.0 - unc_3)

        # Mean delta from most specific available level
        mean_delta = 0.0
        if p3 is not None and p3.n_observations >= self.min_evidence:
            mean_delta = p3.mean_delta
        elif p2 is not None and p2.n_observations >= self.min_evidence:
            mean_delta = p2.mean_delta
        elif p1 is not None and p1.n_observations >= self.min_evidence:
            mean_delta = p1.mean_delta
        else:
            mean_delta = p0.mean_delta

        return PosteriorQueryResult(
            success_probability=est_3,
            uncertainty=unc_3,
            confidence=confidence,
            mean_delta=mean_delta,
            primary_level=primary_level,
            evidence_counts={
                3: p3.n_observations if p3 else 0,
                2: p2.n_observations if p2 else 0,
                1: p1.n_observations if p1 else 0,
                0: p0.n_observations,
            },
            raw_posteriors={
                3: p3.to_dict() if p3 else None,
                2: p2.to_dict() if p2 else None,
                1: p1.to_dict() if p1 else None,
                0: p0.to_dict(),
            },
        )

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics for diagnostics."""
        return {
            "n_level3_contexts": len(self._level3),
            "n_level2_contexts": len(self._level2),
            "n_level1_contexts": len(self._level1),
            "global_observations": self._global.n_observations,
            "global_mean": round(self._global.mean, 4),
            "global_uncertainty": round(self._global.uncertainty, 4),
            "kappa": self.kappa,
            "min_evidence": self.min_evidence,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire posterior memory for persistence."""
        return {
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "kappa": self.kappa,
            "min_evidence": self.min_evidence,
            "global": self._global.to_dict(),
            "level1": {str(k): v.to_dict() for k, v in self._level1.items()},
            "level2": {str(k): v.to_dict() for k, v in self._level2.items()},
            "level3": {str(k): v.to_dict() for k, v in self._level3.items()},
        }


@dataclass
class PosteriorQueryResult:
    """Result of a hierarchical posterior query.

    Provides everything the contextual policy needs to score an action:
    success probability, uncertainty, confidence, and diagnostic provenance.
    """
    success_probability: float    # Smoothed P(success|action, context)
    uncertainty: float            # Posterior uncertainty in [0, 1]
    confidence: float             # 1 - uncertainty
    mean_delta: float             # Expected metric improvement
    primary_level: int            # Which context level dominated (3=specific, 0=global)
    evidence_counts: Dict[int, int] = field(default_factory=dict)
    raw_posteriors: Dict[int, Optional[Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success_probability": round(self.success_probability, 6),
            "uncertainty": round(self.uncertainty, 6),
            "confidence": round(self.confidence, 6),
            "mean_delta": round(self.mean_delta, 6),
            "primary_level": self.primary_level,
            "evidence_counts": dict(self.evidence_counts),
        }
