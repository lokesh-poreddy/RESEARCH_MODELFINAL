"""researchforge/transfer/utility_learner.py — Phase 12B Transfer Utility Learner.

Models the conditional transfer utility E[Delta | Cs, Ct, Cst, M, a] using a 
hierarchical Bayesian approach to handle sparse observations across distinct 
research topologies, yielding operational probabilities for Benefit, Harm, and Neutral.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple
import math
from scipy.stats import norm

class TransferUtilityPredictor:
    """Online Bayesian utility predictor for counterfactual transfer deltas.
    
    Maintains a Gaussian posterior over the weights of a linear contextual model,
    allowing for sequential updates and uncertainty-aware predictions of P_B, P_N, P_H.
    """
    
    def __init__(self, feature_dim: int, alpha: float = 1.0, beta: float = 1.0, delta_threshold: float = 0.005):
        """
        Args:
            feature_dim: Dimensionality of the context feature vector.
            alpha: Prior precision (1/variance) of the weights.
            beta: Precision (1/variance) of the observation noise.
            delta_threshold: Threshold for classifying Benefit (> +d) vs Harm (< -d).
        """
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.beta = beta
        self.delta_threshold = delta_threshold
        
        # Posterior over weights w ~ N(mean_w, cov_w)
        self.mean_w = np.zeros(feature_dim)
        self.cov_w = np.eye(feature_dim) / alpha
        
        # Cache precision matrix (inverse covariance) for fast updates
        self.prec_w = np.eye(feature_dim) * alpha

    def _extract_features(self, context_features: Dict[str, Any]) -> np.ndarray:
        """Flattens the structured 9-part context dictionary into a fixed-length vector."""
        if 'source_context' in context_features:
            x = np.zeros(self.feature_dim)
            x[0] = 1.0  # bias
            
            # C_t
            tc = context_features.get('target_context', {})
            x[1] = tc.get('current_metric', 0.0)
            
            # C_st
            rc = context_features.get('relationship_context', {})
            x[2] = float(rc.get('cross_family', 0))
            
            # M
            ms = context_features.get('memory_state', {})
            x[3] = ms.get('posterior_mean', 0.0)
            x[4] = ms.get('posterior_variance', 1.0)
            
            # E
            eq = context_features.get('evidence_quality', {})
            x[5] = float(eq.get('observation_count', 0))
            
            # U
            unc = context_features.get('uncertainty', {})
            x[6] = unc.get('epistemic', 1.0)
            
            # F
            fs = context_features.get('failure_state', {})
            x[7] = fs.get('recent_failure_rate', 0.0)
            
            # R
            rt = context_features.get('research_trajectory', {})
            x[8] = float(rt.get('budget_remaining', 0))
            
            return x

        if 'x' in context_features:
            x = np.array(context_features['x'], dtype=float)
            # pad or truncate if needed
            if len(x) < self.feature_dim:
                x = np.pad(x, (0, self.feature_dim - len(x)))
            return x[:self.feature_dim]
        
        # Fallback
        x = np.zeros(self.feature_dim)
        x[0] = 1.0 # bias
        return x
        return x

    def predict_utility(self, context_features: Dict[str, Any]) -> Tuple[float, float, Dict[str, float]]:
        """Predict expected utility E[Delta] and variance Var[Delta].
        
        Returns:
            pred_mean: E[Delta | x]
            pred_var: Var(Delta | x)
            probabilities: {'P_B': float, 'P_N': float, 'P_H': float}
        """
        x = self._extract_features(context_features)
        
        # Predictive mean: E[y*] = x^T mu
        pred_mean = float(np.dot(x, self.mean_w))
        
        # Predictive variance: Var(y*) = 1/beta + x^T Cov x
        epistemic_var = float(np.dot(x, np.dot(self.cov_w, x)))
        pred_var = (1.0 / self.beta) + epistemic_var
        
        # Calculate probabilities P(Delta > d), P(Delta < -d)
        std_dev = math.sqrt(pred_var)
        if std_dev < 1e-6:
            # Degenerate case, step function
            P_B = 1.0 if pred_mean > self.delta_threshold else 0.0
            P_H = 1.0 if pred_mean < -self.delta_threshold else 0.0
            P_N = 1.0 - P_B - P_H
        else:
            # P_B = 1 - CDF(delta_threshold)
            P_B = 1.0 - norm.cdf(self.delta_threshold, loc=pred_mean, scale=std_dev)
            # P_H = CDF(-delta_threshold)
            P_H = norm.cdf(-self.delta_threshold, loc=pred_mean, scale=std_dev)
            
            # Bound numerically
            P_B = max(0.0, min(1.0, P_B))
            P_H = max(0.0, min(1.0, P_H))
            P_N = max(0.0, 1.0 - (P_B + P_H))

        probabilities = {
            "P_B": P_B,
            "P_N": P_N,
            "P_H": P_H
        }
        
        return pred_mean, pred_var, probabilities

    def update(self, context_features: Dict[str, Any], observed_delta: float) -> None:
        """Online Bayesian update given an observed counterfactual Delta_t."""
        x = self._extract_features(context_features)
        
        # Precision update: S_new = S_old + beta * (x x^T)
        self.prec_w += self.beta * np.outer(x, x)
        
        # Covariance update (Sherman-Morrison could be used for speed, 
        # but explicit inverse is fine for low D)
        self.cov_w = np.linalg.inv(self.prec_w)
        
        # Mean update: mu_new = Cov_new * (S_old * mu_old + beta * y * x)
        # equivalently: mu_new = mu_old + beta * Cov_new * x * (y - x^T mu_old)
        error = observed_delta - np.dot(x, self.mean_w)
        self.mean_w += self.beta * error * np.dot(self.cov_w, x)

    def get_predictor_callable(self) -> Callable[[Dict[str, Any]], Tuple[float, float]]:
        """Returns a callable compatible with CounterfactualEvidenceService."""
        def predictor(ctx: Dict[str, Any]) -> Tuple[float, float]:
            mean, var, _ = self.predict_utility(ctx)
            return mean, var
        return predictor
