"""researchforge/policy/saturation.py — Research saturation detection and pivot recommendations.

Scientific Invariant (Phase 9):
Saturation does NOT mean failure, and must NOT automatically terminate research.
A saturated or stagnant research region indicates that the current strategy family or
representation is exhausted, triggering pivot recommendations (e.g. gather evidence,
change strategy, change representation).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..domain.base import DomainObject
from ..domain.claim import Claim
from ..domain.outcome import Outcome


class SaturationState(str, Enum):
    ACTIVE = "ACTIVE"
    DIMINISHING_RETURNS = "DIMINISHING_RETURNS"
    SATURATED = "SATURATED"
    CONTRADICTORY = "CONTRADICTORY"
    STAGNANT = "STAGNANT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class PivotRecommendation(str, Enum):
    CONTINUE = "CONTINUE"
    REPLICATE = "REPLICATE"
    GATHER_EVIDENCE = "GATHER_EVIDENCE"
    CHANGE_STRATEGY = "CHANGE_STRATEGY"
    CHANGE_REPRESENTATION = "CHANGE_REPRESENTATION"
    CHANGE_MODEL_FAMILY = "CHANGE_MODEL_FAMILY"
    REQUEST_HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"


@dataclass(frozen=True)
class SaturationReport(DomainObject):
    """Detailed diagnostic report on current research saturation status."""
    current_status: SaturationState
    recommendation: PivotRecommendation
    signal_values: Dict[str, float]
    thresholds: Dict[str, float]
    triggering_evidence: List[str]
    rationale: str
    evaluated_at: float = field(default_factory=time.time)
    provenance_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "SaturationReport":
        d = dict(obj)
        if "current_status" in d and isinstance(d["current_status"], str):
            d["current_status"] = SaturationState(d["current_status"])
        if "recommendation" in d and isinstance(d["recommendation"], str):
            d["recommendation"] = PivotRecommendation(d["recommendation"])
        return cls(**d)


class ResearchSaturationDetector:
    """Evaluates empirical trajectories to identify plateaus, diminishing returns, and contradictions."""

    def __init__(
        self,
        min_samples: int = 3,
        diminishing_returns_delta: float = 0.005,
        stagnant_variance_threshold: float = 0.0001,
        window_size: int = 5,
    ) -> None:
        self.min_samples = min_samples
        self.diminishing_returns_delta = diminishing_returns_delta
        self.stagnant_variance_threshold = stagnant_variance_threshold
        self.window_size = window_size

    def evaluate(
        self,
        recent_outcomes: List[Outcome],
        active_claims: Optional[List[Claim]] = None,
        unresolved_contradictions: Optional[List[str]] = None,
    ) -> SaturationReport:
        """Evaluate trajectory to determine saturation state and pivot recommendation."""
        thresholds = {
            "min_samples": float(self.min_samples),
            "diminishing_returns_delta": self.diminishing_returns_delta,
            "stagnant_variance_threshold": self.stagnant_variance_threshold,
        }

        # 1. Check for contradictions first
        contradiction_count = len(unresolved_contradictions or [])
        if active_claims:
            for c in active_claims:
                if c.contradicting_evidence_ids and c.supporting_evidence_ids:
                    contradiction_count += 1

        if contradiction_count > 0:
            return SaturationReport(
                id=f"sat_report_{int(time.time())}",
                schema_version="1.0",
                current_status=SaturationState.CONTRADICTORY,
                recommendation=PivotRecommendation.REPLICATE,
                signal_values={"contradiction_count": float(contradiction_count)},
                thresholds=thresholds,
                triggering_evidence=unresolved_contradictions or [],
                rationale="Unresolved contradictory evidence detected across experiments. Recommend replication or evidence gathering.",
            )

        # 2. Check evidence sufficiency
        valid_metrics = [
            o.measured_metrics.get("metric", 0.0)
            for o in recent_outcomes
            if o.measured_metrics and "metric" in o.measured_metrics
        ]
        if len(valid_metrics) < self.min_samples:
            return SaturationReport(
                id=f"sat_report_{int(time.time())}",
                schema_version="1.0",
                current_status=SaturationState.INSUFFICIENT_EVIDENCE,
                recommendation=PivotRecommendation.GATHER_EVIDENCE,
                signal_values={"sample_count": float(len(valid_metrics))},
                thresholds=thresholds,
                triggering_evidence=[o.id for o in recent_outcomes],
                rationale=f"Insufficient samples ({len(valid_metrics)} < {self.min_samples}) to evaluate saturation.",
            )

        # Take last window
        window = valid_metrics[-self.window_size:]
        n = len(window)
        mean_val = sum(window) / n
        variance = sum((x - mean_val) ** 2 for x in window) / n
        recent_deltas = [window[i] - window[i - 1] for i in range(1, n)]
        avg_delta = sum(recent_deltas) / len(recent_deltas) if recent_deltas else 0.0

        signal_values = {
            "sample_count": float(n),
            "recent_mean": round(mean_val, 4),
            "variance": round(variance, 6),
            "recent_average_delta": round(avg_delta, 6),
        }

        # 3. Check for stagnation (near-zero variance, no progress)
        if variance < self.stagnant_variance_threshold and abs(avg_delta) < 0.001:
            return SaturationReport(
                id=f"sat_report_{int(time.time())}",
                schema_version="1.0",
                current_status=SaturationState.STAGNANT,
                recommendation=PivotRecommendation.CHANGE_STRATEGY,
                signal_values=signal_values,
                thresholds=thresholds,
                triggering_evidence=[o.id for o in recent_outcomes[-n:]],
                rationale="Variance across recent outcomes is below stagnation threshold. Recommend changing strategy family.",
            )

        # 4. Check for diminishing returns or saturation
        if avg_delta <= 0.0:
            return SaturationReport(
                id=f"sat_report_{int(time.time())}",
                schema_version="1.0",
                current_status=SaturationState.SATURATED,
                recommendation=PivotRecommendation.CHANGE_REPRESENTATION,
                signal_values=signal_values,
                thresholds=thresholds,
                triggering_evidence=[o.id for o in recent_outcomes[-n:]],
                rationale="Non-positive average delta across recent evaluations indicates plateau. Recommend representation pivot.",
            )
        elif avg_delta < self.diminishing_returns_delta:
            return SaturationReport(
                id=f"sat_report_{int(time.time())}",
                schema_version="1.0",
                current_status=SaturationState.DIMINISHING_RETURNS,
                recommendation=PivotRecommendation.CHANGE_MODEL_FAMILY,
                signal_values=signal_values,
                thresholds=thresholds,
                triggering_evidence=[o.id for o in recent_outcomes[-n:]],
                rationale=f"Recent average gain ({avg_delta:.4f}) is below diminishing returns threshold ({self.diminishing_returns_delta}).",
            )

        # 5. Otherwise actively progressing
        return SaturationReport(
            id=f"sat_report_{int(time.time())}",
            schema_version="1.0",
            current_status=SaturationState.ACTIVE,
            recommendation=PivotRecommendation.CONTINUE,
            signal_values=signal_values,
            thresholds=thresholds,
            triggering_evidence=[o.id for o in recent_outcomes[-n:]],
            rationale="Research trajectory is actively improving above diminishing returns threshold.",
        )
