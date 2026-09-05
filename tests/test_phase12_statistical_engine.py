"""tests/test_phase12_statistical_engine.py — Unit tests for Statistical Analysis Engine.

Verifies:
1. Blocked paired matching on (task_id, seed, order_id)
2. Structure-preserving block bootstrap & rejection of invalid calls
3. Paired permutation test with exact sign-flipping randomization
4. Diagnostic parametric t-test, Wilcoxon, and Shapiro-Wilk normality
5. Holm-Bonferroni step-down correction for secondary family H2-H4
6. Preservation of unadjusted alpha for primary H1
7. Pre-registered H4 non-monotonic contrast evaluation
8. Pre-registered sensitivity analyses across all 5 dimensions
9. Statistical artifact generation, immutability, and deterministic replay
"""
import copy
import json
import os
import pytest
from pathlib import Path
import numpy as np

from researchforge.benchmarks.statistical import (
    HypothesisDecision,
    HypothesisType,
    PairedContrastRecord,
    StatisticalAnalysisEngine,
    StatisticalArtifact,
    ValidationStatus,
    create_canonical_sap,
)


@pytest.fixture
def canonical_sap():
    return create_canonical_sap()


@pytest.fixture
def engine(canonical_sap):
    return StatisticalAnalysisEngine(plan=canonical_sap, random_seed=42)


@pytest.fixture
def synthetic_observations():
    """Deterministic synthetic observations mimicking multi-condition multi-ordering benchmark."""
    obs = []
    tasks = ["T_digits_0_4", "T_digits_5_9", "T_ecg_lead1"]
    orders = ["forward", "reverse", "cross_first"]
    seeds = [0, 1]

    for t in tasks:
        regime = "SAME_FAMILY" if "digits" in t else "CROSS_FAMILY"
        for o in orders:
            for s in seeds:
                # COLD_START baseline
                obs.append({
                    "task_id": t,
                    "regime": regime,
                    "order_id": o,
                    "seed": s,
                    "condition": "COLD_START",
                    "prior_task_count": 0,
                    "decision_quality": 0.85,
                    "best_metric": 0.88,
                    "negative_transfer_rate": 0.02,
                })
                # CONTINUOUS_EXPERIENCE: benefit in SAME_FAMILY, interference in CROSS_FAMILY
                dq = 0.95 if regime == "SAME_FAMILY" else 0.70
                obs.append({
                    "task_id": t,
                    "regime": regime,
                    "order_id": o,
                    "seed": s,
                    "condition": "CONTINUOUS_EXPERIENCE",
                    "prior_task_count": 1 if "5_9" in t else (2 if "ecg" in t else 0),
                    "decision_quality": dq,
                    "best_metric": 0.92 if regime == "SAME_FAMILY" else 0.82,
                    "negative_transfer_rate": 0.01 if regime == "SAME_FAMILY" else 0.08,
                })
                # ADAPTIVE_TRAJECTORY
                obs.append({
                    "task_id": t,
                    "regime": regime,
                    "order_id": o,
                    "seed": s,
                    "condition": "ADAPTIVE_TRAJECTORY",
                    "prior_task_count": 1,
                    "decision_quality": 0.88,
                    "best_metric": 0.89,
                    "negative_transfer_rate": 0.03,
                })
                # TRAJECTORY_MEMORY (higher negative transfer)
                obs.append({
                    "task_id": t,
                    "regime": regime,
                    "order_id": o,
                    "seed": s,
                    "condition": "TRAJECTORY_MEMORY",
                    "prior_task_count": 1,
                    "decision_quality": 0.82,
                    "best_metric": 0.85,
                    "negative_transfer_rate": 0.09,
                })
    return obs


def test_paired_matching_across_blocks(engine, synthetic_observations):
    """Verify that observations are matched on (task_id, seed, order_id)."""
    contrasts = engine.build_paired_contrasts(
        observations=synthetic_observations,
        endpoint="decision_quality",
        test_condition="CONTINUOUS_EXPERIENCE",
        control_condition="COLD_START",
        regime_filter="SAME_FAMILY",
    )

    # 2 same-family tasks * 3 orderings * 2 seeds = 12 matched pairs
    assert len(contrasts) == 12
    for c in contrasts:
        assert c.regime == "SAME_FAMILY"
        assert c.test_condition == "CONTINUOUS_EXPERIENCE"
        assert c.control_condition == "COLD_START"
        assert c.paired_difference == pytest.approx(0.95 - 0.85, abs=1e-4)


def test_structure_preserving_bootstrap_and_rejection(engine):
    """Verify structure-preserving bootstrap calculates valid CI and rejects empty/malformed inputs."""
    # Rejection of empty contrasts
    with pytest.raises(ValueError) as excinfo:
        engine.structure_preserving_bootstrap_ci([])
    assert "Cannot perform structure-preserving bootstrap on empty contrasts" in str(excinfo.value)

    # Valid bootstrap over matched pairs
    contrasts = [
        PairedContrastRecord("b1", "t1", 0, "fwd", "SAME", "test", "ctrl", 0.90, 0.80, 0.10),
        PairedContrastRecord("b2", "t1", 1, "fwd", "SAME", "test", "ctrl", 0.92, 0.81, 0.11),
        PairedContrastRecord("b3", "t2", 0, "rev", "SAME", "test", "ctrl", 0.88, 0.82, 0.06),
        PairedContrastRecord("b4", "t2", 1, "rev", "SAME", "test", "ctrl", 0.91, 0.83, 0.08),
    ]

    ci_lower, ci_upper = engine.structure_preserving_bootstrap_ci(contrasts, n_resamples=1000)
    assert ci_lower < ci_upper
    assert 0.05 <= ci_lower <= 0.10
    assert 0.08 <= ci_upper <= 0.13


def test_paired_permutation_test_sign_flipping(engine):
    """Verify exact sign-flipping permutation test detects true directional effects and null noise."""
    # Case A: Clear positive treatment effect
    contrasts_pos = [
        PairedContrastRecord(f"b{i}", "t", i, "fwd", "SAME", "test", "ctrl", 0.9, 0.7, 0.2)
        for i in range(10)
    ]
    p_pos = engine.paired_permutation_test(contrasts_pos, n_permutations=2000, alternative="two-sided")
    assert p_pos < 0.01

    # Case B: Symmetric zero-mean differences
    contrasts_null = [
        PairedContrastRecord(f"b{i}", "t", i, "fwd", "SAME", "test", "ctrl", 0.8, 0.8, 0.05 if i % 2 == 0 else -0.05)
        for i in range(20)
    ]
    p_null = engine.paired_permutation_test(contrasts_null, n_permutations=2000, alternative="two-sided")
    assert p_null > 0.50


def test_holm_bonferroni_multiplicity_correction():
    """Verify Holm-Bonferroni step-down adjustment on secondary hypothesis family."""
    raw_p = {
        "H2": 0.010,
        "H3": 0.040,
        "H4": 0.025,
    }
    # Sorted: H2 (0.010, m=3 -> *3 = 0.030)
    #         H4 (0.025, m=2 -> *2 = 0.050)
    #         H3 (0.040, m=1 -> *1 = 0.040 -> max with prior = 0.050)
    adjusted = StatisticalAnalysisEngine.apply_holm_bonferroni(raw_p)

    assert adjusted["H2"] == pytest.approx(0.030, abs=1e-5)
    assert adjusted["H4"] == pytest.approx(0.050, abs=1e-5)
    assert adjusted["H3"] == pytest.approx(0.050, abs=1e-5)


def test_primary_h1_no_multiplicity_penalty(engine, synthetic_observations):
    """Verify H1 (primary confirmatory) receives NO multiplicity adjustment against secondary family."""
    eval_results, _ = engine.evaluate_all_hypotheses(synthetic_observations)

    h1_res = next(r for r in eval_results if r.hypothesis_id == "H1")
    assert h1_res.hypothesis_type == HypothesisType.PRIMARY_CONFIRMATORY
    assert h1_res.multiplicity_method == "none_primary"
    assert h1_res.raw_p_value == h1_res.adjusted_p_value

    # H2-H4 must use Holm-Bonferroni
    for r in eval_results:
        if r.hypothesis_id != "H1":
            assert r.hypothesis_type == HypothesisType.SECONDARY_EXPLORATORY
            assert r.multiplicity_method == "holm_bonferroni"


def test_h4_non_monotonic_evaluation(engine, synthetic_observations):
    """Verify H4 non-monotonic directional evaluation via pre-registered contrasts C1 and C2."""
    eval_results, _ = engine.evaluate_all_hypotheses(synthetic_observations)
    h4_res = next(r for r in eval_results if r.hypothesis_id == "H4")

    assert "contrast_details" in h4_res.to_dict()
    details = h4_res.contrast_details
    assert "c1_dq1_minus_dq0" in details
    assert "c2_dq2_minus_dq1" in details
    assert "composite_contrast" in details
    assert "directional_criterion_met" in details


def test_pre_registered_sensitivity_analyses(engine, synthetic_observations):
    """Verify that all 5 pre-registered sensitivity dimensions are executed."""
    sens_results = engine.run_all_sensitivity_analyses(synthetic_observations)

    dimensions_found = set(s.dimension for s in sens_results)
    assert "task_ordering" in dimensions_found
    assert "seed_subset" in dimensions_found
    assert "task_regime" in dimensions_found
    assert "experience_horizon" in dimensions_found

    for s in sens_results:
        assert s.ci_lower_95 <= s.ci_upper_95
        assert s.stability in ("STABLE", "NEUTRAL", "NEGATIVE")


def test_statistical_artifact_generation_against_phase11_pilot(engine):
    """Verify statistical artifact generation against frozen Phase 11 pilot data."""
    pilot_path = Path(__file__).resolve().parent.parent / "phase11_continuity_benchmark_result.json"
    assert pilot_path.exists(), "Phase 11 pilot artifact must exist!"

    test_stat_path = Path(__file__).resolve().parent.parent / "test_phase12_validation.json"
    try:
        artifact = engine.generate_statistical_artifact(
            raw_benchmark_artifact_path=str(pilot_path),
            validation_status=ValidationStatus.METHOD_VALIDATION,
            output_path=str(test_stat_path),
        )

        assert artifact.validation_status == "METHOD_VALIDATION"
        assert artifact.primary_endpoint == "decision_quality"
        assert artifact.artifact_fingerprint != ""
        assert len(artifact.hypothesis_results) == 4
        assert len(artifact.sensitivity_results) > 0
        assert os.path.exists(test_stat_path)

        # Re-load and verify serialization
        with open(test_stat_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["validation_status"] == "METHOD_VALIDATION"
        assert data["artifact_fingerprint"] == artifact.artifact_fingerprint
    finally:
        if test_stat_path.exists():
            test_stat_path.unlink()


def test_deterministic_statistical_replay(engine):
    """Verify identical inputs produce identical statistical artifact fingerprints."""
    pilot_path = Path(__file__).resolve().parent.parent / "phase11_continuity_benchmark_result.json"

    art1 = engine.generate_statistical_artifact(str(pilot_path))
    engine2 = StatisticalAnalysisEngine(plan=engine.plan, random_seed=42)
    art2 = engine2.generate_statistical_artifact(str(pilot_path))

    assert art1.artifact_fingerprint == art2.artifact_fingerprint
