"""researchforge/transfer/utility_learner.py — Phase 12B Transfer Utility Learner.

Models the conditional transfer utility E[Delta | Cs, Ct, Cst, M, a] using a
hierarchical Bayesian approach to handle sparse observations across distinct
research topologies, yielding operational probabilities for Benefit, Harm, and Neutral.

Mathematical Foundations:
  - Bayesian Linear Regression with conjugate Normal-Normal model.
  - Online sequential updates via Sherman-Morrison or explicit matrix update.
  - Predictive distribution: p(y*|x*) = N(x*^T mu_w, sigma^2 + x*^T Cov_w x*)
  - P_B = 1 - Phi((d / sigma_predictive)), P_H = Phi((-d) / sigma_predictive)
"""
from __future__ import annotations

import math
import numpy as np
from typing import Any, Callable, Dict, Tuple

from scipy.stats import norm


class TransferUtilityPredictor:
    """Online Bayesian utility predictor for counterfactual transfer deltas.

    Maintains a Gaussian posterior over the weights of a linear contextual model,
    allowing for sequential updates and uncertainty-aware predictions of P_B, P_N, P_H.

    Feature Vector (9-dim, index-named):
      [0] bias=1.0
      [1] target_current_metric      ∈ [0,1]
      [2] cross_family               ∈ {0,1}
      [3] posterior_mean             ∈ [0,1]
      [4] posterior_variance         ∈ [0,1] (normalised)
      [5] log1p(observation_count)   ∈ [0, ~log(100)]  (log-normalised)
      [6] epistemic_uncertainty      ∈ [0,1]
      [7] recent_failure_rate        ∈ [0,1]
      [8] budget_remaining           ∈ [0,1] (normalised fraction)
    """

    # Expected feature vector length — hard-coded for schema stability
    FEATURE_DIM: int = 9

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        alpha: float = 1.0,
        beta: float = 10.0,   # Higher observation precision → faster learning
        delta_threshold: float = 0.005,
    ) -> None:
        """
        Args:
            feature_dim: Dimensionality of the context feature vector.
            alpha: Prior precision (1/prior_variance) over the weight vector.
            beta: Likelihood precision (1/noise_variance) of observed deltas.
            delta_threshold: Absolute delta threshold for Benefit (> +d) and Harm (< -d).
        """
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta = beta
        self.delta_threshold = delta_threshold

        # Posterior over weights: w ~ N(mean_w, cov_w)
        self.mean_w: np.ndarray = np.zeros(feature_dim)
        self.cov_w: np.ndarray = np.eye(feature_dim) / alpha

        # Precision matrix (inverse of covariance) — used for sequential update
        self.prec_w: np.ndarray = np.eye(feature_dim) * alpha

        # Telemetry
        self.n_updates: int = 0
        self.observed_deltas: list[float] = []

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------

    def _extract_features(self, context_features: Dict[str, Any]) -> np.ndarray:
        """Flattens the structured 9-part context dict into a fixed-length vector.

        All values are normalised to ensure no single feature dominates the
        posterior update due to scale differences.
        """
        x = np.zeros(self.feature_dim)
        x[0] = 1.0  # bias term

        if "source_context" in context_features or "target_context" in context_features:
            # --- Target context ---
            tc = context_features.get("target_context", {})
            x[1] = float(tc.get("current_metric", 0.0))  # already ∈ [0,1]

            # --- Relationship context ---
            rc = context_features.get("relationship_context", {})
            x[2] = 1.0 if rc.get("cross_family", False) else 0.0

            # --- Memory state ---
            ms = context_features.get("memory_state", {})
            x[3] = float(ms.get("posterior_mean", 0.0))  # ∈ [0,1]
            raw_var = float(ms.get("posterior_variance", 1.0))
            x[4] = min(raw_var, 1.0)  # clip to [0,1]

            # --- Evidence quality: log-normalise count ---
            eq = context_features.get("evidence_quality", {})
            obs_count = max(0, int(eq.get("observation_count", 0)))
            x[5] = math.log1p(obs_count) / math.log1p(100)  # normalised ∈ [0,1]

            # --- Uncertainty ---
            unc = context_features.get("uncertainty", {})
            x[6] = float(unc.get("epistemic", 1.0))  # ∈ [0,1]

            # --- Failure state ---
            fs = context_features.get("failure_state", {})
            x[7] = float(fs.get("recent_failure_rate", 0.0))  # ∈ [0,1]

            # --- Research trajectory: normalise budget fraction ---
            rt = context_features.get("research_trajectory", {})
            x[8] = float(rt.get("budget_remaining", 0.0))  # ∈ [0,1] already

            return x

        # Legacy flat-vector fallback
        if "x" in context_features:
            raw = np.array(context_features["x"], dtype=float)
            if len(raw) < self.feature_dim:
                raw = np.pad(raw, (0, self.feature_dim - len(raw)))
            return raw[: self.feature_dim]

        # Uninformative fallback (bias only)
        return x

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_utility(
        self, context_features: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, float]]:
        """Predict expected utility E[Delta] and variance Var[Delta].

        Returns:
            pred_mean: E[Delta | x]
            pred_var:  Var(Delta | x)  — includes both epistemic and aleatoric
            probabilities: {'P_B': float, 'P_N': float, 'P_H': float}
        """
        x = self._extract_features(context_features)

        # Predictive mean: E[y*] = x^T mu
        pred_mean = float(np.dot(x, self.mean_w))

        # Predictive variance: Var(y*) = 1/beta + x^T Cov_w x
        epistemic_var = float(x @ self.cov_w @ x)
        pred_var = (1.0 / self.beta) + epistemic_var

        # Classify into P_B, P_N, P_H via Normal CDF
        std_dev = math.sqrt(max(pred_var, 1e-12))

        P_B = 1.0 - norm.cdf(self.delta_threshold, loc=pred_mean, scale=std_dev)
        P_H = norm.cdf(-self.delta_threshold, loc=pred_mean, scale=std_dev)

        # Clamp for numerical safety
        P_B = float(np.clip(P_B, 0.0, 1.0))
        P_H = float(np.clip(P_H, 0.0, 1.0))
        P_N = float(np.clip(1.0 - (P_B + P_H), 0.0, 1.0))

        return pred_mean, pred_var, {"P_B": P_B, "P_N": P_N, "P_H": P_H}

    # ------------------------------------------------------------------
    # Online Update (Bayesian Linear Regression — conjugate update)
    # ------------------------------------------------------------------

    def update(self, context_features: Dict[str, Any], observed_delta: float) -> None:
        """Sequential Bayesian update given an observed counterfactual Delta_t.

        Uses the standard conjugate Normal-Normal update:
          S_new  = S_old + beta * (x x^T)
          mu_new = Cov_new * (S_old * mu_old + beta * y * x)

        Numerically equivalent to the recursive formula:
          mu_new = mu_old + beta * Cov_new * x * (y - x^T mu_old)
        """
        x = self._extract_features(context_features)

        # Precision update
        self.prec_w = self.prec_w + self.beta * np.outer(x, x)

        # Covariance update (full inverse; feature_dim=9 so this is cheap)
        self.cov_w = np.linalg.inv(self.prec_w)

        # Mean update via the innovation (residual) form
        innovation = observed_delta - float(np.dot(x, self.mean_w))
        self.mean_w = self.mean_w + self.beta * innovation * (self.cov_w @ x)

        # Telemetry
        self.n_updates += 1
        self.observed_deltas.append(float(observed_delta))

    # ------------------------------------------------------------------
    # Interoperability
    # ------------------------------------------------------------------

    def get_predictor_callable(self) -> Callable[[Dict[str, Any]], Tuple[float, float]]:
        """Returns a callable compatible with CounterfactualEvidenceService."""

        def predictor(ctx: Dict[str, Any]) -> Tuple[float, float]:
            mean, var, _ = self.predict_utility(ctx)
            return mean, var

        return predictor

    def get_stats(self) -> Dict[str, Any]:
        """Returns telemetry about the predictor's learning history."""
        deltas = self.observed_deltas
        return {
            "n_updates": self.n_updates,
            "mean_observed_delta": float(np.mean(deltas)) if deltas else 0.0,
            "std_observed_delta": float(np.std(deltas)) if len(deltas) > 1 else 0.0,
            "weight_norm": float(np.linalg.norm(self.mean_w)),
        }
