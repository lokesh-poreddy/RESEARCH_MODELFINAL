"""tests/test_phase13_transfer_guard.py — TransferGuard and Selective Gating tests.

Classification: CORE
Verifies:
1. Prior affinity distinguishes structural characteristics without claiming empirical truth.
2. Low prior affinity is NEVER treated as proof of negative transfer.
3. STRICT mode produces clean cold-start isolation.
4. SELECTIVE mode filters by typed knowledge category.
5. False-positive protection: positive historical empirical effect permits cross-family transfer.
"""
import pytest
from researchforge.transfer.affinity import (
    TaskProfile,
    compute_historical_transfer_effect,
    compute_prior_affinity,
    compute_transfer_score,
)
from researchforge.transfer.guard import TransferGuard
from researchforge.transfer.types import (
    ALLOWED_SELECTIVE_CATEGORIES,
    RESTRICTED_SELECTIVE_CATEGORIES,
    TransferGuardMode,
    TransferableKnowledgeCategory,
)


@pytest.fixture
def digits_profile() -> TaskProfile:
    return TaskProfile(
        task_id="digits_0_4",
        task_family="DIGITS_SPATIAL",
        modality="spatial",
        n_features=64,
        n_classes=5,
    )


@pytest.fixture
def digits_other_profile() -> TaskProfile:
    return TaskProfile(
        task_id="digits_5_9",
        task_family="DIGITS_SPATIAL",
        modality="spatial",
        n_features=64,
        n_classes=5,
    )


@pytest.fixture
def ecg_profile() -> TaskProfile:
    return TaskProfile(
        task_id="synthetic_ecg_lead1",
        task_family="ECG_TEMPORAL",
        modality="temporal",
        n_features=100,
        n_classes=2,
    )


@pytest.fixture
def xor_profile() -> TaskProfile:
    return TaskProfile(
        task_id="tabular_xor_parity",
        task_family="XOR_TABULAR",
        modality="tabular",
        n_features=10,
        n_classes=2,
    )


def test_prior_affinity_same_vs_cross_family(digits_profile, digits_other_profile, ecg_profile):
    """Prior affinity should be higher within same family than across families."""
    aff_same = compute_prior_affinity(digits_profile, digits_other_profile)
    aff_cross = compute_prior_affinity(digits_profile, ecg_profile)

    assert aff_same > aff_cross
    assert aff_same >= 0.85
    assert aff_cross < 0.50


def test_low_prior_affinity_not_treated_as_negative_transfer(digits_profile, ecg_profile):
    """Low prior affinity with zero evidence should produce a cautious neutral score, NOT negative transfer."""
    aff_cross = compute_prior_affinity(digits_profile, ecg_profile)
    score_no_evidence = compute_transfer_score(
        prior_affinity=aff_cross,
        historical_transfer_effect=None,
        evidence_volume=0,
    )
    # Must NOT be severe negative transfer (e.g. < -0.3)
    assert score_no_evidence >= -0.20
    assert score_no_evidence <= 0.20


def test_strict_mode_yields_clean_isolation(digits_profile, digits_other_profile):
    """STRICT mode should suppress all memories regardless of family relatedness."""
    guard = TransferGuard(mode=TransferGuardMode.STRICT)
    memories = [
        {"id": "mem1", "text": "heuristic 1", "knowledge_category": "RESEARCH_PROCEDURE"},
        {"id": "mem2", "text": "heuristic 2", "knowledge_category": "GENERIC_FAILURE_PATTERN"},
    ]

    filtered, dec = guard.filter_memories(memories, digits_profile, digits_other_profile)
    assert len(filtered) == 0
    assert dec.action_taken == "SUPPRESS_ALL"
    assert len(dec.allowed_memory_ids) == 0
    assert len(dec.filtered_memory_ids) == 2


def test_selective_mode_filters_by_knowledge_category(digits_profile, xor_profile):
    """In cross-family transfer under SELECTIVE mode, portable knowledge is allowed, task assumptions restricted."""
    guard = TransferGuard(mode=TransferGuardMode.SELECTIVE)
    memories = [
        {"id": "proc_1", "text": "decrease capacity on divergence", "knowledge_category": TransferableKnowledgeCategory.RESEARCH_PROCEDURE},
        {"id": "fail_1", "text": "overfitting warning on low data", "knowledge_category": TransferableKnowledgeCategory.GENERIC_FAILURE_PATTERN},
        {"id": "hyp_1", "text": "8x8 conv filter best for digits", "knowledge_category": TransferableKnowledgeCategory.TASK_SPECIFIC_HYPOTHESIS},
        {"id": "feat_1", "text": "assume 64 pixel features", "knowledge_category": TransferableKnowledgeCategory.FEATURE_ASSUMPTIONS},
    ]

    filtered, dec = guard.filter_memories(memories, digits_profile, xor_profile)
    allowed_ids = {m["id"] for m in filtered}

    assert allowed_ids == {"proc_1", "fail_1"}
    assert "hyp_1" not in allowed_ids
    assert "feat_1" not in allowed_ids
    assert dec.action_taken == "FILTER_TYPED"


def test_false_positive_suppression_protection(digits_profile, ecg_profile):
    """If empirical historical transfer effect is positive with high confidence, cross-family transfer is allowed."""
    guard = TransferGuard(mode=TransferGuardMode.SELECTIVE, false_positive_protection=True)
    memories = [
        {"id": "hyp_cross", "text": "model config", "knowledge_category": TransferableKnowledgeCategory.TASK_SPECIFIC_HYPOTHESIS},
    ]
    # Simulated empirical audits showing positive transfer from digits to ecg
    history = [
        {"source_family": "DIGITS_SPATIAL", "target_family": "ECG_TEMPORAL", "decision_quality_delta": 0.25},
        {"source_family": "DIGITS_SPATIAL", "target_family": "ECG_TEMPORAL", "decision_quality_delta": 0.18},
        {"source_family": "DIGITS_SPATIAL", "target_family": "ECG_TEMPORAL", "decision_quality_delta": 0.22},
    ]

    filtered, dec = guard.filter_memories(memories, digits_profile, ecg_profile, transfer_history=history)

    assert len(filtered) == 1
    assert dec.action_taken == "ALLOW_EMPIRICAL_EXCEPTION"
    assert "False-positive protection" in dec.rationale
