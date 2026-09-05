"""researchforge/benchmarks/statistical/models.py — Statistical Evaluation contracts.

Phase 12A Invariants:
1. Primary confirmatory endpoint (DQ) has NO multiplicity adjustment against secondary family.
2. Secondary family (H2-H4) uses pre-registered Holm-Bonferroni correction.
3. Block-aware matched contrasts preserve experimental structure (task, seed, order).
4. Strict separation between METHOD_VALIDATION and CONFIRMATORY_EXECUTION.
5. Content-addressed SHA-256 artifact fingerprinting.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ...domain.base import DomainObject, _canonical_json


class EndpointType(str, Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    EXPLORATORY = "EXPLORATORY"


class HypothesisType(str, Enum):
    PRIMARY_CONFIRMATORY = "PRIMARY_CONFIRMATORY"
    SECONDARY_EXPLORATORY = "SECONDARY_EXPLORATORY"


class ValidationStatus(str, Enum):
    METHOD_VALIDATION = "METHOD_VALIDATION"          # Executed against pilot data to test calculation machinery
    CONFIRMATORY_EXECUTION = "CONFIRMATORY_EXECUTION"  # Executed against full pre-registered dataset
    EXPLORATORY_ANALYSIS = "EXPLORATORY_ANALYSIS"      # Post-hoc sensitivity or observational extension


class HypothesisDecision(str, Enum):
    REJECT_NULL = "REJECT_NULL"
    FAIL_TO_REJECT = "FAIL_TO_REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class PairedContrastRecord:
    """Matched contrast between test and control conditions within a single experimental block."""
    block_id: str             # Deterministic identifier: f"{task_id}_{seed}_{order_id}"
    task_id: str
    seed: int
    order_id: str
    regime: str
    test_condition: str
    control_condition: str
    test_value: float
    control_value: float
    paired_difference: float   # test_value - control_value
    prior_task_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "order_id": self.order_id,
            "regime": self.regime,
            "test_condition": self.test_condition,
            "control_condition": self.control_condition,
            "test_value": round(self.test_value, 4),
            "control_value": round(self.control_value, 4),
            "paired_difference": round(self.paired_difference, 4),
            "prior_task_count": self.prior_task_count,
        }


@dataclass
class PairedTestResult:
    """Comprehensive paired estimation and hypothesis testing results across matched blocks."""
    endpoint: str
    comparison_name: str
    test_condition: str
    control_condition: str
    regime_filter: Optional[str]
    n_pairs: int

    # Parametric & non-parametric point estimates
    mean_difference: float
    median_difference: float
    std_difference: float

    # Uncertainty: Block-aware bootstrap 95% Confidence Interval
    ci_lower_95: float
    ci_upper_95: float

    # Standardized effect sizes
    cohen_dz: float
    rank_biserial_r: float

    # Statistical significance: Primary inference via paired permutation/randomization
    permutation_p_value: float
    wilcoxon_p_value: Optional[float] = None
    paired_t_p_value: Optional[float] = None

    # Diagnostic normality check on paired differences
    normality_shapiro_p: Optional[float] = None
    is_normal: bool = False

    # Blocked bootstrap parameters
    bootstrap_resamples: int = 2000
    resampling_unit: str = "matched_experimental_block"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "comparison_name": self.comparison_name,
            "test_condition": self.test_condition,
            "control_condition": self.control_condition,
            "regime_filter": self.regime_filter or "ALL",
            "n_pairs": self.n_pairs,
            "mean_difference": round(self.mean_difference, 4),
            "median_difference": round(self.median_difference, 4),
            "std_difference": round(self.std_difference, 4),
            "ci_lower_95": round(self.ci_lower_95, 4),
            "ci_upper_95": round(self.ci_upper_95, 4),
            "cohen_dz": round(self.cohen_dz, 4),
            "rank_biserial_r": round(self.rank_biserial_r, 4),
            "permutation_p_value": round(self.permutation_p_value, 6),
            "wilcoxon_p_value": round(self.wilcoxon_p_value, 6) if self.wilcoxon_p_value is not None else None,
            "paired_t_p_value": round(self.paired_t_p_value, 6) if self.paired_t_p_value is not None else None,
            "normality_shapiro_p": round(self.normality_shapiro_p, 4) if self.normality_shapiro_p is not None else None,
            "is_normal": self.is_normal,
            "bootstrap_resamples": self.bootstrap_resamples,
            "resampling_unit": self.resampling_unit,
        }


@dataclass
class HypothesisEvaluationResult:
    """Pre-registered hypothesis evaluation record with pre/post multiplicity adjustment."""
    hypothesis_id: str               # "H1", "H2", "H3", "H4"
    hypothesis_type: HypothesisType  # PRIMARY_CONFIRMATORY vs SECONDARY_EXPLORATORY
    description: str
    endpoint: str
    raw_p_value: float
    adjusted_p_value: float          # Raw for H1; Holm-Bonferroni adjusted for H2-H4
    multiplicity_method: str         # "none_primary" vs "holm_bonferroni"
    effect_size: float
    effect_metric: str               # e.g., "cohen_dz", "rank_biserial_r", "mean_contrast"
    ci_95: Tuple[float, float]
    decision: HypothesisDecision
    contrast_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_type": self.hypothesis_type.value,
            "description": self.description,
            "endpoint": self.endpoint,
            "raw_p_value": round(self.raw_p_value, 6),
            "adjusted_p_value": round(self.adjusted_p_value, 6),
            "multiplicity_method": self.multiplicity_method,
            "effect_size": round(self.effect_size, 4),
            "effect_metric": self.effect_metric,
            "ci_95": [round(self.ci_95[0], 4), round(self.ci_95[1], 4)],
            "decision": self.decision.value,
            "contrast_details": self.contrast_details,
        }


@dataclass
class SensitivityResult:
    """Outcome of sensitivity analysis along a pre-registered design axis."""
    dimension: str     # "task_ordering", "seed_subset", "task_regime", "experience_horizon", "transfer_epsilon"
    stratum: str       # e.g., "forward", "seed_0", "SAME_FAMILY", "horizon_1", "epsilon_0.005"
    endpoint: str
    n_pairs: int
    mean_difference: float
    ci_lower_95: float
    ci_upper_95: float
    stability: str     # "STABLE" (CI excludes 0 in expected direction), "NEUTRAL", "INVERTED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "stratum": self.stratum,
            "endpoint": self.endpoint,
            "n_pairs": self.n_pairs,
            "mean_difference": round(self.mean_difference, 4),
            "ci_lower_95": round(self.ci_lower_95, 4),
            "ci_upper_95": round(self.ci_upper_95, 4),
            "stability": self.stability,
        }


@dataclass
class StatisticalArtifact:
    """Canonical versioned statistical artifact generated independently from raw benchmark data."""
    artifact_version: str = "1.0.0"
    validation_status: str = ValidationStatus.METHOD_VALIDATION.value
    plan_version: str = "1.0.0"
    plan_fingerprint: str = ""
    code_revision: str = "HEAD"
    raw_benchmark_artifact_path: str = ""
    raw_benchmark_fingerprint: str = ""

    primary_endpoint: str = "decision_quality"
    secondary_endpoints: List[str] = field(default_factory=list)

    hypothesis_results: List[Dict[str, Any]] = field(default_factory=list)
    paired_test_results: List[Dict[str, Any]] = field(default_factory=list)
    sensitivity_results: List[Dict[str, Any]] = field(default_factory=list)
    missing_data_report: Dict[str, Any] = field(default_factory=dict)
    statistical_software: Dict[str, str] = field(default_factory=dict)

    artifact_fingerprint: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_fingerprint(self) -> str:
        payload = {
            "artifact_version": self.artifact_version,
            "validation_status": self.validation_status,
            "plan_version": self.plan_version,
            "plan_fingerprint": self.plan_fingerprint,
            "raw_benchmark_fingerprint": self.raw_benchmark_fingerprint,
            "primary_endpoint": self.primary_endpoint,
            "hypothesis_results": self.hypothesis_results,
            "sensitivity_count": len(self.sensitivity_results),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        fp = self.artifact_fingerprint or self.compute_fingerprint()
        return {
            "artifact_version": self.artifact_version,
            "validation_status": self.validation_status,
            "plan_version": self.plan_version,
            "plan_fingerprint": self.plan_fingerprint,
            "code_revision": self.code_revision,
            "raw_benchmark_artifact_path": self.raw_benchmark_artifact_path,
            "raw_benchmark_fingerprint": self.raw_benchmark_fingerprint,
            "primary_endpoint": self.primary_endpoint,
            "secondary_endpoints": self.secondary_endpoints,
            "hypothesis_results": self.hypothesis_results,
            "paired_test_results": self.paired_test_results,
            "sensitivity_results": self.sensitivity_results,
            "missing_data_report": self.missing_data_report,
            "statistical_software": self.statistical_software,
            "artifact_fingerprint": fp,
            "created_at": self.created_at,
        }
