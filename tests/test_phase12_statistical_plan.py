"""tests/test_phase12_statistical_plan.py — Unit tests for Statistical Analysis Plan (SAP).

Verifies:
1. Immutability of sealed StatisticalAnalysisPlan (PlanMutationError on modification)
2. Hypothesis definitions (H1 primary confirmatory, H2-H4 secondary exploratory)
3. Endpoint hierarchy: primary Decision Quality, explicit secondary endpoints
4. Resampling unit specification (matched_experimental_block)
5. Holm-Bonferroni specification for secondary family with no penalty for primary H1
6. Pre-registered sensitivity dimensions (all 5 axes present)
7. Missing data policy and exclusion policy definitions
8. Canonical fingerprint determinism
"""
import copy
import pytest

from researchforge.benchmarks.statistical import (
    HypothesisType,
    PlanMutationError,
    StatisticalAnalysisPlan,
    create_canonical_sap,
)


def test_canonical_sap_creation_and_immutability():
    """Verify that canonical SAP is properly constructed, sealed, and immutable."""
    sap = create_canonical_sap()

    assert sap.is_sealed is True
    assert sap.plan_version == "1.0.0"
    assert sap.plan_fingerprint != ""
    assert len(sap.plan_fingerprint) == 64

    # Attempting to mutate sealed plan must raise PlanMutationError
    with pytest.raises(PlanMutationError) as excinfo:
        sap.primary_endpoint = "best_metric"
    assert "sealed and immutable" in str(excinfo.value)

    with pytest.raises(PlanMutationError):
        sap.alpha_primary = 0.01

    with pytest.raises(PlanMutationError):
        sap.hypotheses["H1"] = None


def test_hypothesis_specifications_and_hierarchy():
    """Verify pre-registered hypotheses H1–H4 and their confirmatory/exploratory status."""
    sap = create_canonical_sap()

    assert "H1" in sap.hypotheses
    assert "H2" in sap.hypotheses
    assert "H3" in sap.hypotheses
    assert "H4" in sap.hypotheses

    # H1 is strictly PRIMARY_CONFIRMATORY
    h1 = sap.hypotheses["H1"]
    assert h1.hypothesis_type == HypothesisType.PRIMARY_CONFIRMATORY
    assert h1.endpoint == "decision_quality"
    assert h1.test_condition == "CONTINUOUS_EXPERIENCE"
    assert h1.control_condition == "COLD_START"
    assert h1.regime_filter == "SAME_FAMILY"
    assert h1.expected_direction == "GREATER"

    # H2 is SECONDARY_EXPLORATORY for cross-family negative transfer
    h2 = sap.hypotheses["H2"]
    assert h2.hypothesis_type == HypothesisType.SECONDARY_EXPLORATORY
    assert h2.regime_filter == "CROSS_FAMILY"
    assert h2.expected_direction == "LESS"

    # H3 is SECONDARY_EXPLORATORY for adaptive retrieval safety
    h3 = sap.hypotheses["H3"]
    assert h3.hypothesis_type == HypothesisType.SECONDARY_EXPLORATORY
    assert h3.test_condition == "ADAPTIVE_TRAJECTORY"
    assert h3.control_condition == "TRAJECTORY_MEMORY"
    assert h3.expected_direction == "LESS"

    # H4 is SECONDARY_EXPLORATORY for non-monotonicity
    h4 = sap.hypotheses["H4"]
    assert h4.hypothesis_type == HypothesisType.SECONDARY_EXPLORATORY
    assert h4.expected_direction == "NON_MONOTONIC"


def test_endpoint_hierarchy_and_multiplicity_rules():
    """Verify single primary endpoint and separation of H1 from secondary multiplicity."""
    sap = create_canonical_sap()

    assert sap.primary_endpoint == "decision_quality"
    assert "best_metric" in sap.secondary_endpoints
    assert "research_efficiency" in sap.secondary_endpoints
    assert "failure_repetition_rate" in sap.secondary_endpoints
    assert "negative_transfer_rate" in sap.secondary_endpoints

    # Invariant (Correction 2): alpha_primary = 0.05 without multiplicity penalty
    assert sap.alpha_primary == 0.05
    assert sap.alpha_secondary_family == 0.05
    assert sap.multiplicity_procedure_secondary == "holm_bonferroni"


def test_h4_contrast_specification():
    """Verify that H4 non-monotonicity is formulated via explicit directional contrasts."""
    sap = create_canonical_sap()

    assert "c1" in sap.h4_contrast_spec
    assert "c2" in sap.h4_contrast_spec
    assert "composite_contrast" in sap.h4_contrast_spec
    assert "C1 > 0 and C2 < 0" in sap.h4_contrast_spec["decision_rule"]


def test_resampling_unit_and_sensitivity_dimensions():
    """Verify structure-preserving resampling unit and all 5 pre-registered sensitivity axes."""
    sap = create_canonical_sap()

    # Invariant (Correction 4): Block-preserving resampling unit
    assert sap.resampling_unit == "matched_experimental_block"
    assert sap.bootstrap_resamples >= 2000
    assert sap.permutation_resamples >= 5000

    expected_dims = [
        "task_ordering",
        "seed_subset",
        "task_regime",
        "experience_horizon",
        "transfer_epsilon",
    ]
    for dim in expected_dims:
        assert dim in sap.sensitivity_dimensions

    assert sap.transfer_epsilon_levels == [0.005, 0.010, 0.020]
    assert sap.missing_data_policy == "complete_case_no_post_hoc_removal"
    assert sap.exclusion_policy == "strictly_pre_registered_no_outlier_removal"


def test_fingerprint_determinism_and_tamper_detection():
    """Verify deterministic fingerprinting and divergence when plan content changes."""
    sap1 = create_canonical_sap()
    sap2 = create_canonical_sap()

    assert sap1.plan_fingerprint == sap2.plan_fingerprint

    # Mutating an unsealed plan changes the fingerprint
    unsealed = StatisticalAnalysisPlan(primary_endpoint="different_endpoint")
    assert unsealed.compute_fingerprint() != sap1.plan_fingerprint
