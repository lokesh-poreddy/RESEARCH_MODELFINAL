"""researchforge/benchmarks/statistical/plan.py — Formal Statistical Analysis Plan (SAP).

Phase 12A Invariants:
1. Pre-registration integrity: Sealed, immutable plan locked before observing large-scale runs.
2. Primary endpoint: Decision Quality (DQ) with no multiplicity adjustment (alpha = 0.05).
3. Secondary family (H2-H4): Holm-Bonferroni multiplicity correction.
4. Resampling unit: Matched experimental block (task_id, seed, order_id).
5. Non-monotonicity (H4): Explicit directional contrasts C1 > 0 and C2 < 0.
6. Sensitivity analyses along 5 pre-registered axes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...domain.base import _canonical_json
from .models import EndpointType, HypothesisType


class PlanMutationError(Exception):
    """Raised when an attempt is made to alter a sealed StatisticalAnalysisPlan."""
    pass


@dataclass(frozen=True)
class HypothesisSpec:
    """Pre-registered hypothesis specification."""
    hypothesis_id: str
    hypothesis_type: HypothesisType
    title: str
    description: str
    endpoint: str
    test_condition: str
    control_condition: str
    regime_filter: Optional[str]
    expected_direction: str  # "GREATER", "LESS", "NON_MONOTONIC"
    contrast_formula: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_type": self.hypothesis_type.value,
            "title": self.title,
            "description": self.description,
            "endpoint": self.endpoint,
            "test_condition": self.test_condition,
            "control_condition": self.control_condition,
            "regime_filter": self.regime_filter,
            "expected_direction": self.expected_direction,
            "contrast_formula": self.contrast_formula,
        }


class FrozenDict(dict):
    """A dictionary that forbids item modification once sealed."""
    def __setitem__(self, key: Any, value: Any) -> None:
        raise PlanMutationError(f"StatisticalAnalysisPlan dictionary is sealed and immutable! Cannot set key '{key}'.")

    def __delitem__(self, key: Any) -> None:
        raise PlanMutationError(f"StatisticalAnalysisPlan dictionary is sealed and immutable! Cannot delete key '{key}'.")

    def clear(self) -> None:
        raise PlanMutationError("StatisticalAnalysisPlan dictionary is sealed and immutable!")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise PlanMutationError("StatisticalAnalysisPlan dictionary is sealed and immutable!")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise PlanMutationError("StatisticalAnalysisPlan dictionary is sealed and immutable!")


@dataclass
class StatisticalAnalysisPlan:
    """Canonical, pre-registered and immutable Statistical Analysis Plan."""
    plan_version: str = "1.0.0"
    title: str = "ResearchForge Research Continuity Benchmark Statistical Analysis Plan"
    code_revision: str = "HEAD"

    # Canonical sign convention: delta = test_value - control_value
    sign_convention: str = "CANONICAL_TEST_MINUS_CONTROL"

    # Endpoints hierarchy
    primary_endpoint: str = "decision_quality"
    secondary_endpoints: List[str] = field(default_factory=lambda: [
        "best_metric",
        "research_efficiency",
        "search_efficiency",
        "failure_repetition_rate",
        "negative_transfer_rate",
        "harmful_transfer_rate",
    ])

    # Inference & Multiplicity control
    confidence_level: float = 0.95
    alpha_primary: float = 0.05
    alpha_secondary_family: float = 0.05
    multiplicity_procedure_secondary: str = "holm_bonferroni"

    # Blocked resampling specification
    resampling_unit: str = "matched_experimental_block"  # (task_id, seed, order_id)
    bootstrap_resamples: int = 2000
    permutation_resamples: int = 5000

    # Pre-registered hypotheses
    hypotheses: Dict[str, HypothesisSpec] = field(default_factory=dict)

    # H4 specific non-monotonic contrast definitions
    h4_contrast_spec: Dict[str, str] = field(default_factory=lambda: {
        "c1": "DQ(prior_task_count=1) - DQ(prior_task_count=0)",
        "c2": "DQ(prior_task_count=2) - DQ(prior_task_count=1)",
        "composite_contrast": "DQ_1 - (DQ_0 + DQ_2) / 2",
        "decision_rule": "H4 supported if C1 > 0 and C2 < 0 under pre-specified contrast",
    })

    # Data integrity & exclusions
    missing_data_policy: str = "complete_case_no_post_hoc_removal"
    exclusion_policy: str = "strictly_pre_registered_no_outlier_removal"

    # Sensitivity analysis dimensions
    sensitivity_dimensions: List[str] = field(default_factory=lambda: [
        "task_ordering",
        "seed_subset",
        "task_regime",
        "experience_horizon",
        "transfer_epsilon",
    ])
    transfer_epsilon_levels: List[float] = field(default_factory=lambda: [0.005, 0.010, 0.020])

    is_sealed: bool = False
    sealed_at: str = ""
    plan_fingerprint: str = ""

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "is_sealed", False) and name not in ("plan_fingerprint",):
            raise PlanMutationError(
                f"StatisticalAnalysisPlan is sealed and immutable! Attempted to modify '{name}'."
            )
        super().__setattr__(name, value)

    def compute_fingerprint(self) -> str:
        payload = {
            "plan_version": self.plan_version,
            "sign_convention": self.sign_convention,
            "primary_endpoint": self.primary_endpoint,
            "secondary_endpoints": self.secondary_endpoints,
            "confidence_level": self.confidence_level,
            "alpha_primary": self.alpha_primary,
            "alpha_secondary_family": self.alpha_secondary_family,
            "multiplicity_procedure_secondary": self.multiplicity_procedure_secondary,
            "resampling_unit": self.resampling_unit,
            "hypotheses": {k: v.to_dict() for k, v in sorted(self.hypotheses.items())},
            "h4_contrast_spec": self.h4_contrast_spec,
            "sensitivity_dimensions": self.sensitivity_dimensions,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def seal(self) -> "StatisticalAnalysisPlan":
        """Seals the analysis plan, preventing any post-hoc modifications."""
        self.sealed_at = datetime.now(timezone.utc).isoformat()
        self.plan_fingerprint = self.compute_fingerprint()
        self.hypotheses = FrozenDict(self.hypotheses)
        self.is_sealed = True
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "title": self.title,
            "code_revision": self.code_revision,
            "sign_convention": self.sign_convention,
            "primary_endpoint": self.primary_endpoint,
            "secondary_endpoints": self.secondary_endpoints,
            "confidence_level": self.confidence_level,
            "alpha_primary": self.alpha_primary,
            "alpha_secondary_family": self.alpha_secondary_family,
            "multiplicity_procedure_secondary": self.multiplicity_procedure_secondary,
            "resampling_unit": self.resampling_unit,
            "bootstrap_resamples": self.bootstrap_resamples,
            "permutation_resamples": self.permutation_resamples,
            "hypotheses": {k: v.to_dict() for k, v in self.hypotheses.items()},
            "h4_contrast_spec": self.h4_contrast_spec,
            "missing_data_policy": self.missing_data_policy,
            "exclusion_policy": self.exclusion_policy,
            "sensitivity_dimensions": self.sensitivity_dimensions,
            "transfer_epsilon_levels": self.transfer_epsilon_levels,
            "is_sealed": self.is_sealed,
            "sealed_at": self.sealed_at,
            "plan_fingerprint": self.plan_fingerprint or self.compute_fingerprint(),
        }


def create_canonical_sap() -> StatisticalAnalysisPlan:
    """Factory creating the canonical, pre-registered Phase 12A Statistical Analysis Plan."""
    h1 = HypothesisSpec(
        hypothesis_id="H1",
        hypothesis_type=HypothesisType.PRIMARY_CONFIRMATORY,
        title="Primary Confirmatory: Same-Family Experience Benefit",
        description="Context-appropriate accumulated research experience improves Decision Quality (DQ) compared with controlled cold-start in pre-specified SAME_FAMILY settings.",
        endpoint="decision_quality",
        test_condition="CONTINUOUS_EXPERIENCE",
        control_condition="COLD_START",
        regime_filter="SAME_FAMILY",
        expected_direction="GREATER",
        contrast_formula="DQ(CONTINUOUS_EXPERIENCE, SAME_FAMILY) - DQ(COLD_START, SAME_FAMILY) > 0",
    )

    h2 = HypothesisSpec(
        hypothesis_id="H2",
        hypothesis_type=HypothesisType.SECONDARY_EXPLORATORY,
        title="Secondary: Cross-Family Negative Transfer",
        description="Cross-family or mismatched accumulated research experience produces negative transfer (degradation in DQ) relative to within-task cold start baseline.",
        endpoint="decision_quality",
        test_condition="CONTINUOUS_EXPERIENCE",
        control_condition="COLD_START",
        regime_filter="CROSS_FAMILY",
        expected_direction="LESS",
        contrast_formula="DQ(CONTINUOUS_EXPERIENCE, CROSS_FAMILY) - DQ(COLD_START, CROSS_FAMILY) < 0",
    )

    h3 = HypothesisSpec(
        hypothesis_id="H3",
        hypothesis_type=HypothesisType.SECONDARY_EXPLORATORY,
        title="Secondary: Adaptive Retrieval Safety",
        description="Context-sensitive adaptive trajectory retrieval with backoff reduces harmful/negative transfer compared with static trajectory memory under sparse or mismatched conditions.",
        endpoint="negative_transfer_rate",
        test_condition="ADAPTIVE_TRAJECTORY",
        control_condition="TRAJECTORY_MEMORY",
        regime_filter=None,
        expected_direction="LESS",
        contrast_formula="NTR(ADAPTIVE_TRAJECTORY) - NTR(TRAJECTORY_MEMORY) < 0",
    )

    h4 = HypothesisSpec(
        hypothesis_id="H4",
        hypothesis_type=HypothesisType.SECONDARY_EXPLORATORY,
        title="Secondary: Non-Monotonic Experience Accumulation",
        description="The benefit of accumulated experience exhibits early within-family gains followed by cross-family saturation/interference across cumulative prior task counts.",
        endpoint="decision_quality",
        test_condition="CONTINUOUS_EXPERIENCE",
        control_condition="COLD_START",
        regime_filter=None,
        expected_direction="NON_MONOTONIC",
        contrast_formula="C1 = DQ_1 - DQ_0 > 0 and C2 = DQ_2 - DQ_1 < 0",
    )

    sap = StatisticalAnalysisPlan(
        hypotheses={
            "H1": h1,
            "H2": h2,
            "H3": h3,
            "H4": h4,
        }
    )
    return sap.seal()
