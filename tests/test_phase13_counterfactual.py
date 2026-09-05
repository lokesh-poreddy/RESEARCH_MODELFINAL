"""tests/test_phase13_counterfactual.py — Tri-state counterfactual arbitration tests.

Classification: CORE
Verifies:
1. COUNTERFACTUAL_OBSERVED when paired executions are provided.
2. COUNTERFACTUAL_ESTIMATED when validated surrogate predictions are provided.
3. COUNTERFACTUAL_UNAVAILABLE when neither exists, verifying the system falls back
   cleanly to evidence-based gating without fabricating counterfactual expectations.
"""
import pytest
from researchforge.transfer.counterfactual import (
    CounterfactualArbitrator,
    CounterfactualEvaluation,
    CounterfactualStatus,
)


def test_counterfactual_observed_paired_execution():
    """Observed paired runs produce exact empirical delta."""
    arbitrator = CounterfactualArbitrator()
    paired = {"quality_prior": 0.85, "quality_fresh": 0.70}
    eval_res = arbitrator.arbitrate(candidate_spec={"model": "MLP"}, paired_evidence=paired)

    assert eval_res.status == CounterfactualStatus.COUNTERFACTUAL_OBSERVED
    assert eval_res.expected_delta == pytest.approx(0.15)
    assert eval_res.variance == 0.0
    assert eval_res.recommendation == "PROCEED_WITH_TRANSFER"


def test_counterfactual_estimated_surrogate_model():
    """Surrogate model produces predicted delta with explicit variance and confidence interval."""
    def mock_surrogate(spec):
        return 0.12, 0.01  # mean=0.12, var=0.01

    arbitrator = CounterfactualArbitrator(surrogate_model=mock_surrogate)
    eval_res = arbitrator.arbitrate(candidate_spec={"model": "RandomForest"})

    assert eval_res.status == CounterfactualStatus.COUNTERFACTUAL_ESTIMATED
    assert eval_res.expected_delta == pytest.approx(0.12)
    assert eval_res.variance == pytest.approx(0.01)
    assert eval_res.confidence_interval is not None
    assert eval_res.confidence_interval[0] < 0.12 < eval_res.confidence_interval[1]
    assert eval_res.recommendation == "PROCEED_WITH_TRANSFER"


def test_counterfactual_unavailable_falls_back_to_evidence_gating():
    """Without paired execution or surrogate model, status is UNAVAILABLE and no expectation is fabricated."""
    arbitrator = CounterfactualArbitrator()
    eval_res = arbitrator.arbitrate(candidate_spec={"model": "SVC"})

    assert eval_res.status == CounterfactualStatus.COUNTERFACTUAL_UNAVAILABLE
    assert eval_res.expected_delta is None
    assert eval_res.variance is None
    assert eval_res.confidence_interval is None
    assert eval_res.recommendation == "FALLBACK_TO_EVIDENCE_GATING"
    assert "cannot be known" in eval_res.rationale
