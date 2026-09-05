"""researchforge/transfer/affinity.py — Task affinity and empirical transfer effect modeling.

Phase 13 (RF-1.0.0-beta.1):
Strictly distinguishes engineering priors (prior_affinity) from empirical
observations (historical_transfer_effect). Low prior affinity must NEVER
be treated as proof of negative transfer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TaskProfile:
    """Structural characteristics of a task used for prior affinity computation."""
    task_id: str
    task_family: str                # e.g. "DIGITS_SPATIAL", "ECG_TEMPORAL", "XOR_TABULAR"
    modality: str                   # "spatial", "temporal", "tabular"
    n_features: int
    n_classes: int
    is_synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_prior_affinity(source: TaskProfile, target: TaskProfile) -> float:
    """Compute engineering prior affinity A(Ts, Tt) in [0.0, 1.0].
    
    This is an architectural heuristic prior based on structural similarity.
    It is NOT scientific proof of transferability and must not be treated as empirical truth.
    """
    if source.task_id == target.task_id:
        return 1.0

    score = 0.0

    # 1. Modality match (weight 0.40)
    if source.modality == target.modality:
        score += 0.40
    elif {source.modality, target.modality} == {"spatial", "tabular"}:
        score += 0.15
    elif {source.modality, target.modality} == {"temporal", "tabular"}:
        score += 0.15

    # 2. Family match (weight 0.30)
    if source.task_family == target.task_family:
        score += 0.30

    # 3. Feature dimensionality log-ratio closeness (weight 0.15)
    f_ratio = min(source.n_features, target.n_features) / max(source.n_features, target.n_features, 1)
    score += 0.15 * f_ratio

    # 4. Class count parity (weight 0.15)
    c_ratio = min(source.n_classes, target.n_classes) / max(source.n_classes, target.n_classes, 1)
    score += 0.15 * c_ratio

    return max(0.0, min(1.0, score))


def compute_historical_transfer_effect(
    transfer_history: List[Dict[str, Any]],
    source_family: str,
    target_family: str,
) -> Tuple[Optional[float], int]:
    """Extract empirical transfer effect from recorded transfer audits.
    
    Returns:
        (mean_transfer_effect, evidence_volume)
        where mean_transfer_effect in [-1.0, 1.0], or None if no prior audits exist.
    """
    effects: List[float] = []
    for audit in transfer_history:
        sf = audit.get("source_family")
        tf = audit.get("target_family")
        if sf == source_family and tf == target_family:
            delta = audit.get("decision_quality_delta") or audit.get("transfer_delta")
            if delta is not None and isinstance(delta, (int, float)):
                effects.append(float(delta))

    if not effects:
        return None, 0

    return sum(effects) / len(effects), len(effects)


def compute_transfer_score(
    prior_affinity: float,
    historical_transfer_effect: Optional[float],
    evidence_volume: int = 0,
    uncertainty_prior: float = 0.5,
    context_specificity: float = 1.0,
) -> float:
    """Blend prior affinity and empirical transfer effect using Bayesian shrinkage.
    
    Score range: [-1.0, 1.0].
    
    Scientific Invariant:
    A low prior_affinity (e.g. 0.15 across spatial -> tabular) with ZERO empirical
    evidence (volume=0) produces a neutral/cautious score near 0.0, NEVER a strongly
    negative score (e.g. -0.8). Low prior affinity is NOT proof of negative transfer.
    """
    # Map prior affinity [0, 1] to prior expected effect [-0.2, 0.5] (cautious neutral prior)
    # Low prior affinity maps to ~ -0.1 to 0.0 (uninformative/slightly cautious, NOT catastrophic)
    prior_effect = -0.1 + (prior_affinity * 0.6)

    if historical_transfer_effect is None or evidence_volume == 0:
        # No empirical evidence: return bounded prior effect weighted by confidence
        return max(-0.2, min(0.5, prior_effect))

    # Bayesian shrinkage toward observed effect as evidence volume increases
    # N_half = 5 observations
    k = 5.0
    weight_obs = evidence_volume / (evidence_volume + k)
    weight_prior = 1.0 - weight_obs

    blended = (weight_prior * prior_effect) + (weight_obs * historical_transfer_effect)
    return max(-1.0, min(1.0, blended * context_specificity))
