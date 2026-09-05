"""tests/test_phase9_saturation.py — Unit and behavioral tests for ResearchSaturationDetector.

Requirements Tested (Phase 9 Constraint 18):
1. Distinguishes all 6 saturation states:
   ACTIVE, DIMINISHING_RETURNS, SATURATED, CONTRADICTORY, STAGNANT, INSUFFICIENT_EVIDENCE.
2. Provides explicit signal values and thresholds in every report.
3. Invariant: Saturation does NOT automatically terminate research; it only recommends next steps.
"""
import pytest
from researchforge.domain.claim import Claim
from researchforge.domain.outcome import Outcome
from researchforge.policy.saturation import (
    PivotRecommendation,
    ResearchSaturationDetector,
    SaturationReport,
    SaturationState,
)


def _make_outcomes(metrics: list[float]) -> list[Outcome]:
    return [
        Outcome(
            id=f"out_{i}",
            schema_version="1.0",
            run_id=f"run_{i}",
            measured_metrics={"metric": m},
        )
        for i, m in enumerate(metrics)
    ]


def test_insufficient_evidence_state():
    detector = ResearchSaturationDetector(min_samples=3)
    outcomes = _make_outcomes([0.80, 0.81])
    report = detector.evaluate(outcomes)

    assert report.current_status == SaturationState.INSUFFICIENT_EVIDENCE
    assert report.recommendation == PivotRecommendation.GATHER_EVIDENCE
    assert report.signal_values["sample_count"] == 2.0
    assert report.thresholds["min_samples"] == 3.0


def test_contradictory_evidence_state():
    detector = ResearchSaturationDetector(min_samples=3)
    outcomes = _make_outcomes([0.80, 0.82, 0.84])
    claim_with_contradiction = Claim(
        id="c1",
        schema_version="1.0",
        hypothesis_id="h1",
        statement="Scaling hidden units improves test accuracy",
        supporting_evidence_ids=["ev_pos_1"],
        contradicting_evidence_ids=["ev_neg_1"],
    )

    report = detector.evaluate(outcomes, active_claims=[claim_with_contradiction])
    assert report.current_status == SaturationState.CONTRADICTORY
    assert report.recommendation == PivotRecommendation.REPLICATE
    assert report.signal_values["contradiction_count"] >= 1.0


def test_stagnant_state():
    detector = ResearchSaturationDetector(
        min_samples=3,
        stagnant_variance_threshold=0.0001,
        window_size=4,
    )
    # 4 outcomes with virtually zero delta and zero variance
    outcomes = _make_outcomes([0.7500, 0.7500, 0.7500, 0.7500])
    report = detector.evaluate(outcomes)

    assert report.current_status == SaturationState.STAGNANT
    assert report.recommendation == PivotRecommendation.CHANGE_STRATEGY
    assert report.signal_values["variance"] < 0.0001


def test_diminishing_returns_state():
    detector = ResearchSaturationDetector(
        min_samples=3,
        diminishing_returns_delta=0.01,
        window_size=4,
    )
    # Small positive gains: avg delta is ~0.002 < 0.01
    outcomes = _make_outcomes([0.800, 0.802, 0.804, 0.806])
    report = detector.evaluate(outcomes)

    assert report.current_status == SaturationState.DIMINISHING_RETURNS
    assert report.recommendation == PivotRecommendation.CHANGE_MODEL_FAMILY
    assert report.signal_values["recent_average_delta"] < 0.01
    assert report.signal_values["recent_average_delta"] > 0.0


def test_saturated_plateau_state():
    detector = ResearchSaturationDetector(min_samples=3, window_size=4)
    # Non-improving or decreasing sequence
    outcomes = _make_outcomes([0.85, 0.85, 0.84, 0.84])
    report = detector.evaluate(outcomes)

    assert report.current_status == SaturationState.SATURATED
    assert report.recommendation == PivotRecommendation.CHANGE_REPRESENTATION
    assert report.signal_values["recent_average_delta"] <= 0.0


def test_active_progressing_state():
    detector = ResearchSaturationDetector(
        min_samples=3,
        diminishing_returns_delta=0.01,
        window_size=4,
    )
    # Strong positive gains: avg delta = 0.03 > 0.01
    outcomes = _make_outcomes([0.70, 0.73, 0.76, 0.79])
    report = detector.evaluate(outcomes)

    assert report.current_status == SaturationState.ACTIVE
    assert report.recommendation == PivotRecommendation.CONTINUE
    assert report.signal_values["recent_average_delta"] >= 0.01


def test_saturation_does_not_terminate_research():
    """Scientific Rule: Saturation does NOT automatically terminate research; it only recommends next steps."""
    detector = ResearchSaturationDetector(min_samples=3)
    outcomes = _make_outcomes([0.80, 0.80, 0.79, 0.79])
    report = detector.evaluate(outcomes)

    assert report.current_status == SaturationState.SATURATED
    # Generates recommendation, but has no termination side-effect
    assert isinstance(report, SaturationReport)
    assert report.recommendation in [
        PivotRecommendation.CHANGE_REPRESENTATION,
        PivotRecommendation.CHANGE_STRATEGY,
        PivotRecommendation.CHANGE_MODEL_FAMILY,
    ]
