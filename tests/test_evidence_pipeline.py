"""tests/test_evidence_pipeline.py — Phase 8 Evidence Pipeline Unification tests.

Verification suite for:
- Evidence construction and multi-dimensional quality
- Claim construction, status assessment, and contradiction preservation
- Supported, contradicted, uncertain, and speculative relations
- Experiment evidence normalization (valid, invalid, inconclusive, failed)
- Negative evidence retention and retrieval
- Trajectory evidence linkage (referencing VRDEG nodes)
- Literature evidence normalization
- Replication evidence normalization
- Provenance resolution and lineage preservation
- Deterministic semantic fingerprinting and deduplication
- JSON schema validation (evidence.schema.json, claim.schema.json)
- Decision-time evidence snapshot and temporal isolation (no retroactive leakage)
- Backward compatibility
"""
from __future__ import annotations

import json
import os
import time
import pytest
import jsonschema

from researchforge.domain.evidence import (
    Evidence,
    EvidenceType,
    EvidenceQuality,
)
from researchforge.domain.claim import (
    Claim,
    ClaimStatus,
    EvidenceRelation,
)
from researchforge.domain.validity import Validity, ValidityVerdict
from researchforge.domain.outcome import Outcome
from researchforge.domain.experiment import ExperimentRun, ExperimentSpec
from researchforge.domain.failure import Failure
from researchforge.evidence.normalizer import EvidenceNormalizer
from researchforge.evidence.store import EvidenceStore
from researchforge.evidence.snapshot import EvidenceSnapshot

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "..", "researchforge", "schemas")


def load_schema(name: str):
    path = os.path.join(SCHEMA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# 1. Evidence Construction & Multi-dimensional Quality
# ==============================================================================

def test_evidence_construction():
    """Evidence supports typed categories, multi-dimensional quality, and provenance links."""
    quality = EvidenceQuality(
        source_reliability=0.9,
        relevance=0.85,
        empirical_support=0.95,
        replication_status="replicated",
        freshness=0.8,
        contradiction_status="none",
        provenance_completeness=1.0,
    )
    ev = Evidence(
        id="ev_test_1",
        schema_version="1.0",
        source="experiment",
        source_id="run_001",
        evidence_type=EvidenceType.EXPERIMENTAL.value,
        source_type="run",
        claim_id="claim_001",
        relation=EvidenceRelation.SUPPORTED_BY.value,
        provenance_id="prov_001",
        quality=quality,
        originating_run_ids=["run_001"],
        created_at=1000.0,
    )
    assert ev.id == "ev_test_1"
    assert ev.evidence_id == "ev_test_1"
    assert ev.evidence_type == "experimental"
    assert ev.quality is not None
    # Check composite score is inspectable and non-magic
    comp = ev.quality.composite_score()
    assert 0.0 < comp <= 1.0
    assert ev.to_dict()["confidence"] == comp


def test_quality_dimensions_independent():
    """Do NOT use a single magic confidence number; all dimensions remain inspectable."""
    q_high = EvidenceQuality(source_reliability=1.0, empirical_support=1.0, relevance=1.0)
    q_contradicted = EvidenceQuality(source_reliability=1.0, empirical_support=1.0, relevance=1.0, contradiction_status="has_contradiction")
    q_failed_rep = EvidenceQuality(source_reliability=1.0, empirical_support=1.0, relevance=1.0, replication_status="failed_replication")

    # Contradiction and failed replication dampen the composite score without erasing underlying metrics
    assert q_contradicted.composite_score() < q_high.composite_score()
    assert q_failed_rep.composite_score() < q_high.composite_score()
    assert q_contradicted.source_reliability == 1.0  # inspectable intact


# ==============================================================================
# 2. Claim Construction & Status Assessment
# ==============================================================================

def test_claim_construction():
    """Claims represent assertions with explicit status."""
    claim = Claim(
        id="claim_transformer_scaling",
        schema_version="1.0",
        statement="Scaling hidden dimension from 128 to 256 improves accuracy by >5%.",
        claim_type="empirical",
        confidence=0.5,
    )
    assert claim.status == ClaimStatus.SPECULATIVE.value
    assert claim.has_contradiction is False
    assert claim.supporting_evidence_ids is None
    assert claim.contradicting_evidence_ids is None


def test_supported_evidence_updates_claim():
    """Evidence with relation=SUPPORTED_BY marks claim as SUPPORTED."""
    claim = Claim(
        id="c1",
        schema_version="1.0",
        statement="Learning rate 0.001 converges faster than 0.01.",
    )
    ev = Evidence(
        id="ev_sup",
        schema_version="1.0",
        source="experiment",
        relation=EvidenceRelation.SUPPORTED_BY.value,
        claim_id="c1",
    )
    updated = claim.add_evidence(ev)
    assert updated.status == ClaimStatus.SUPPORTED.value
    assert "ev_sup" in (updated.supporting_evidence_ids or [])
    assert not updated.has_contradiction


def test_contradicted_evidence_updates_claim():
    """Evidence with relation=CONTRADICTED_BY marks claim as CONTRADICTED."""
    claim = Claim(
        id="c2",
        schema_version="1.0",
        statement="Increasing batch size to 2048 improves accuracy.",
    )
    ev = Evidence(
        id="ev_contra",
        schema_version="1.0",
        source="experiment",
        relation=EvidenceRelation.CONTRADICTED_BY.value,
        claim_id="c2",
    )
    updated = claim.add_evidence(ev)
    assert updated.status == ClaimStatus.CONTRADICTED.value
    assert "ev_contra" in (updated.contradicting_evidence_ids or [])
    assert not updated.has_contradiction


def test_uncertain_evidence_updates_claim():
    """Evidence with relation=UNCERTAIN_FROM marks claim as UNCERTAIN."""
    claim = Claim(
        id="c3",
        schema_version="1.0",
        statement="Dropout 0.3 outperforms Dropout 0.5.",
    )
    ev = Evidence(
        id="ev_unc",
        schema_version="1.0",
        source="experiment",
        relation=EvidenceRelation.UNCERTAIN_FROM.value,
        claim_id="c3",
    )
    updated = claim.add_evidence(ev)
    assert updated.status == ClaimStatus.UNCERTAIN.value


def test_speculative_evidence_updates_claim():
    """Evidence with relation=SPECULATIVE_FROM marks claim as SPECULATIVE."""
    claim = Claim(
        id="c4",
        schema_version="1.0",
        statement="Quantum kernel improves accuracy on MNIST.",
    )
    ev = Evidence(
        id="ev_spec",
        schema_version="1.0",
        source="hypothesis",
        relation=EvidenceRelation.SPECULATIVE_FROM.value,
        claim_id="c4",
    )
    updated = claim.add_evidence(ev)
    assert updated.status == ClaimStatus.SPECULATIVE.value


# ==============================================================================
# 3. Contradiction Preservation (Rule 8: Do Not Average Contradictions Away)
# ==============================================================================

def test_contradiction_preservation():
    """When both supporting and contradicting evidence exist, both are preserved."""
    claim = Claim(
        id="c_conflict",
        schema_version="1.0",
        statement="AdamW converges strictly faster than SGD across all seeds.",
    )
    ev_sup = Evidence(
        id="ev_seed_1",
        schema_version="1.0",
        source="experiment",
        relation=EvidenceRelation.SUPPORTED_BY.value,
        claim_id="c_conflict",
    )
    ev_contra = Evidence(
        id="ev_seed_2",
        schema_version="1.0",
        source="experiment",
        relation=EvidenceRelation.CONTRADICTED_BY.value,
        claim_id="c_conflict",
    )
    c1 = claim.add_evidence(ev_sup)
    c2 = c1.add_evidence(ev_contra)

    # Both must be preserved!
    assert c2.has_contradiction is True
    assert "ev_seed_1" in (c2.supporting_evidence_ids or [])
    assert "ev_seed_2" in (c2.contradicting_evidence_ids or [])
    assert c2.status == ClaimStatus.UNCERTAIN.value


# ==============================================================================
# 4. Experimental Evidence Normalization
# ==============================================================================

def test_normalize_valid_experiment():
    """Valid experiment outcome normalizes to SUPPORTED_BY evidence with provenance."""
    spec = ExperimentSpec(
        id="spec_01",
        schema_version="1.0",
        rsg_id="rsg_01",
        tmg_id="tmg_01",
        dataset_id="ds_01",
    )
    run = ExperimentRun(
        id="run_01",
        schema_version="1.0",
        spec_id="spec_01",
        start_time="2026-09-05T00:00:00Z",
    )
    outcome = Outcome(
        id="out_01",
        schema_version="1.0",
        run_id="run_01",
        measured_metrics={"accuracy": 0.92, "loss": 0.15},
        validity=Validity(verdict=ValidityVerdict.PASS, details={"rationale": "All assertions passed"}),
    )

    ev = EvidenceNormalizer.normalize_experiment(
        spec=spec,
        run=run,
        outcome=outcome,
        claim_id="claim_acc",
        target_metric="accuracy",
        threshold=0.90,
    )
    assert ev.evidence_type == EvidenceType.EXPERIMENTAL.value
    assert ev.relation == EvidenceRelation.SUPPORTED_BY.value
    assert ev.quality.empirical_support > 0.9
    assert ev.originating_run_ids == ["run_01"]
    assert ev.originating_experiment_ids == ["spec_01"]
    assert ev.originating_outcome_ids == ["out_01"]


def test_normalize_invalid_experiment():
    """Scientific Rule: An invalid experiment MUST NOT become strong positive evidence."""
    spec = ExperimentSpec(
        id="spec_02",
        schema_version="1.0",
        rsg_id="rsg_01",
        tmg_id="tmg_01",
        dataset_id="ds_01",
    )
    run = ExperimentRun(
        id="run_02",
        schema_version="1.0",
        spec_id="spec_02",
        start_time="2026-09-05T00:00:00Z",
    )
    outcome = Outcome(
        id="out_02",
        schema_version="1.0",
        run_id="run_02",
        measured_metrics={"accuracy": 0.99},  # High metric, but INVALID!
        validity=Validity(verdict=ValidityVerdict.FAIL, details={"rationale": "Data leakage detected across train/test splits"}),
    )

    ev = EvidenceNormalizer.normalize_experiment(
        spec=spec,
        run=run,
        outcome=outcome,
        claim_id="claim_leak",
        target_metric="accuracy",
        threshold=0.90,
    )
    # Must NOT be SUPPORTED_BY!
    assert ev.relation == EvidenceRelation.CONTRADICTED_BY.value
    assert ev.quality.empirical_support == 0.0
    assert "Invalid experiment" in (ev.snippet or "")


def test_normalize_inconclusive_experiment():
    """Scientific Rule: Inconclusive outcomes must remain inconclusive."""
    spec = ExperimentSpec(
        id="spec_03",
        schema_version="1.0",
        rsg_id="rsg_01",
        tmg_id="tmg_01",
        dataset_id="ds_01",
    )
    run = ExperimentRun(
        id="run_03",
        schema_version="1.0",
        spec_id="spec_03",
        start_time="2026-09-05T00:00:00Z",
    )
    outcome = Outcome(
        id="out_03",
        schema_version="1.0",
        run_id="run_03",
        measured_metrics={"accuracy": 0.85},
        validity=Validity(verdict=ValidityVerdict.INCONCLUSIVE, details={"rationale": "Sample size too small for statistical significance"}),
    )

    ev = EvidenceNormalizer.normalize_experiment(
        spec=spec,
        run=run,
        outcome=outcome,
        claim_id="claim_inc",
    )
    assert ev.relation == EvidenceRelation.UNCERTAIN_FROM.value
    assert ev.quality.empirical_support == 0.5


def test_normalize_failed_execution_run():
    """Scientific Rule: Execution failure must not automatically produce positive evidence."""
    spec = ExperimentSpec(
        id="spec_04",
        schema_version="1.0",
        rsg_id="rsg_01",
        tmg_id="tmg_01",
        dataset_id="ds_01",
    )
    run = ExperimentRun(
        id="run_04",
        schema_version="1.0",
        spec_id="spec_04",
        start_time="2026-09-05T00:00:00Z",
        failure_info={"message": "CUDA OOM: Out of Memory during forward pass"},
    )

    ev = EvidenceNormalizer.normalize_experiment(
        spec=spec,
        run=run,
        outcome=None,
        claim_id="claim_oom",
    )
    assert ev.relation == EvidenceRelation.CONTRADICTED_BY.value
    assert ev.quality.empirical_support == 0.0
    assert "CUDA OOM" in (ev.snippet or "")


def test_normalize_negative_result():
    """Negative evidence is valuable and preserved as a first-class citizen."""
    ev = EvidenceNormalizer.normalize_negative_result(
        source_id="run_negative_1",
        claim_id="claim_larger_lr",
        finding_summary="Learning rate 0.1 diverged immediately with NaN loss.",
        metrics={"loss": float("nan"), "diverged": True},
        originating_run_ids=["run_negative_1"],
    )
    assert ev.evidence_type == EvidenceType.NEGATIVE.value
    assert ev.relation == EvidenceRelation.CONTRADICTED_BY.value
    assert ev.originating_run_ids == ["run_negative_1"]
    assert "diverged" in (ev.snippet or "")


# ==============================================================================
# 5. Trajectory & Literature Evidence Normalization
# ==============================================================================

def test_normalize_trajectory_evidence():
    """Trajectory evidence references VRDEG nodes without opaque unstructured dumping."""
    ev = EvidenceNormalizer.normalize_trajectory(
        trajectory_id="traj_opt_1",
        node_ids=["node_spec_1", "node_run_1", "node_out_1"],
        trajectory_summary="Optimizer search transitioned from Adam to AdamW with weight decay 0.01.",
        claim_id="claim_adamw",
        relation=EvidenceRelation.SUPPORTED_BY,
        success=True,
    )
    assert ev.evidence_type == EvidenceType.TRAJECTORY.value
    assert "node_spec_1" in ev.originating_vrdeg_node_ids
    assert "node_run_1" in ev.originating_vrdeg_node_ids
    assert "node_out_1" in ev.originating_vrdeg_node_ids
    assert ev.originating_trajectory_ids == ["traj_opt_1"]
    assert ev.relation == EvidenceRelation.SUPPORTED_BY.value


def test_normalize_literature_evidence():
    """Literature findings are normalized with source citation and relevance."""
    ev = EvidenceNormalizer.normalize_literature(
        source_id="doi:10.1000/182",
        title="Attention Is All You Need",
        summary="Transformers replace recurrent models using multi-head self-attention.",
        claim_id="claim_self_attention",
        citation_key="vaswani2017attention",
        doi_or_url="https://doi.org/10.1000/182",
        relevance=0.95,
        relation=EvidenceRelation.SUPPORTED_BY,
    )
    assert ev.evidence_type == EvidenceType.LITERATURE.value
    assert ev.quality.relevance == 0.95
    assert ev.metadata["citation_key"] == "vaswani2017attention"


def test_normalize_replication():
    """Replication status is explicitly tracked; failed replication reduces quality."""
    ev_success = EvidenceNormalizer.normalize_replication(
        original_evidence_id="ev_base",
        replication_run_id="run_rep_1",
        claim_id="claim_rep",
        replicated_successfully=True,
        metrics_comparison={"original_acc": 0.90, "replicated_acc": 0.91},
    )
    assert ev_success.quality.replication_status == "replicated"
    assert ev_success.relation == EvidenceRelation.SUPPORTED_BY.value

    ev_fail = EvidenceNormalizer.normalize_replication(
        original_evidence_id="ev_base",
        replication_run_id="run_rep_2",
        claim_id="claim_rep",
        replicated_successfully=False,
        metrics_comparison={"original_acc": 0.90, "replicated_acc": 0.72},
    )
    assert ev_fail.quality.replication_status == "failed_replication"
    assert ev_fail.relation == EvidenceRelation.CONTRADICTED_BY.value
    assert ev_fail.quality.composite_score() < ev_success.quality.composite_score()


# ==============================================================================
# 6. Deduplication & Semantic Identity
# ==============================================================================

def test_duplicate_detection_and_identity():
    """Equivalent evidence records are deduplicated without collapsing distinct observations."""
    store = EvidenceStore()
    ev1 = EvidenceNormalizer.normalize_literature(
        source_id="arxiv:1706.03762",
        title="Attention Is All You Need",
        summary="Transformers replace recurrence.",
        claim_id="claim_tx",
        doi_or_url="https://arxiv.org/abs/1706.03762",
    )
    # Identical semantic record
    ev2 = EvidenceNormalizer.normalize_literature(
        source_id="arxiv:1706.03762",
        title="Attention Is All You Need",
        summary="Transformers replace recurrence.",
        claim_id="claim_tx",
        doi_or_url="https://arxiv.org/abs/1706.03762",
    )
    # Distinct observation (different source ID and claim)
    ev3 = EvidenceNormalizer.normalize_literature(
        source_id="arxiv:2005.14165",
        title="Language Models are Few-Shot Learners",
        summary="GPT-3 demonstrates in-context learning.",
        claim_id="claim_few_shot",
        doi_or_url="https://arxiv.org/abs/2005.14165",
    )

    added1 = store.add_evidence(ev1)
    added2 = store.add_evidence(ev2)
    added3 = store.add_evidence(ev3)

    assert added1 is True
    assert added2 is False  # Deduplicated!
    assert added3 is True   # Distinct observation accepted!
    assert len(store) == 2


def test_deterministic_fingerprint():
    """Identical evidence produces identical fingerprints; field changes alter fingerprint."""
    ev1 = Evidence(id="e1", schema_version="1.0", source="lab", relation="SUPPORTED_BY")
    ev2 = Evidence(id="e1", schema_version="1.0", source="lab", relation="SUPPORTED_BY")
    ev3 = Evidence(id="e1", schema_version="1.0", source="lab", relation="CONTRADICTED_BY")

    assert ev1.fingerprint() == ev2.fingerprint()
    assert ev1.fingerprint() != ev3.fingerprint()


# ==============================================================================
# 7. Schema Validation & Serialization Roundtrip
# ==============================================================================

def test_evidence_schema_validation():
    """Evidence instances validate against researchforge/schemas/evidence.schema.json."""
    schema = load_schema("evidence.schema.json")
    ev = Evidence(
        id="ev_schema_test",
        schema_version="1.0",
        source="experiment",
        source_id="run_100",
        evidence_type=EvidenceType.EXPERIMENTAL.value,
        source_type="run",
        claim_id="claim_100",
        relation=EvidenceRelation.SUPPORTED_BY.value,
        provenance_id="prov_100",
        quality=EvidenceQuality(source_reliability=0.9, empirical_support=0.95),
        originating_run_ids=["run_100"],
        originating_experiment_ids=["spec_100"],
        created_at=123456.78,
    )
    d = ev.to_dict()
    jsonschema.validate(instance=d, schema=schema)


def test_claim_schema_validation():
    """Claim instances validate against researchforge/schemas/claim.schema.json."""
    schema = load_schema("claim.schema.json")
    claim = Claim(
        id="claim_schema_test",
        schema_version="1.0",
        statement="ResNet with residual connections trains with less vanishing gradient.",
        claim_type="architectural",
        status=ClaimStatus.SUPPORTED.value,
        supporting_evidence_ids=["ev_resnet_1"],
        contradicting_evidence_ids=[],
        confidence=0.9,
    )
    d = claim.to_dict()
    jsonschema.validate(instance=d, schema=schema)


def test_serialization_roundtrips():
    """Evidence and Claim roundtrip via to_dict and from_dict deterministically."""
    ev = Evidence(
        id="ev_rt",
        schema_version="1.0",
        source="lab",
        quality=EvidenceQuality(empirical_support=0.88),
        relation="SUPPORTED_BY",
        created_at=100.0,
    )
    d = ev.to_dict()
    ev_restored = Evidence.from_dict(d)
    assert ev_restored.id == ev.id
    assert ev_restored.quality.empirical_support == 0.88

    claim = Claim(
        id="c_rt",
        schema_version="1.0",
        statement="Scaling test",
        status="SUPPORTED",
        supporting_evidence_ids=["ev_rt"],
    )
    c_d = claim.to_dict()
    c_restored = Claim.from_dict(c_d)
    assert c_restored.id == claim.id
    assert c_restored.supporting_evidence_ids == ["ev_rt"]


# ==============================================================================
# 8. Decision-Time Evidence Snapshot & Temporal Isolation
# ==============================================================================

def test_decision_time_snapshot_temporal_isolation():
    """A later-added evidence item must NOT retroactively appear in an earlier snapshot.
    
    Prevents causal temporal leakage for future ResearchPolicy.
    """
    store = EvidenceStore()
    t_decision = 1000.0

    # Evidence available BEFORE decision
    ev_past_1 = Evidence(
        id="ev_past_1",
        schema_version="1.0",
        source="experiment",
        created_at=900.0,
        relation=EvidenceRelation.SUPPORTED_BY.value,
    )
    ev_past_2 = Evidence(
        id="ev_past_2",
        schema_version="1.0",
        source="literature",
        created_at=999.0,
        relation=EvidenceRelation.SUPPORTED_BY.value,
    )
    # Evidence produced AFTER decision
    ev_future = Evidence(
        id="ev_future",
        schema_version="1.0",
        source="experiment",
        created_at=1050.0,
        relation=EvidenceRelation.CONTRADICTED_BY.value,
    )

    store.add_evidence(ev_past_1)
    store.add_evidence(ev_past_2)
    store.add_evidence(ev_future)

    # Take snapshot as of t_decision = 1000.0
    snapshot = store.create_decision_snapshot(decision_id="decision_D1", as_of_timestamp=t_decision)

    # Past evidence must be present
    assert snapshot.decision_id == "decision_D1"
    assert snapshot.as_of_timestamp == 1000.0
    assert snapshot.evidence_count == 2
    assert "ev_past_1" in snapshot.evidence_ids
    assert "ev_past_2" in snapshot.evidence_ids

    # Future evidence must NOT leak into the snapshot!
    assert "ev_future" not in snapshot.evidence_ids
    assert not snapshot.includes_evidence("ev_future")
    assert snapshot.evidence_items[0].created_at <= 1000.0
    assert snapshot.evidence_items[1].created_at <= 1000.0


def test_snapshot_deterministic_fingerprint():
    """Snapshots are content-addressed and have deterministic fingerprints."""
    ev = Evidence(id="e1", schema_version="1.0", source="s", created_at=50.0)
    store1 = EvidenceStore()
    store1.add_evidence(ev)
    snap1 = store1.create_decision_snapshot("d1", as_of_timestamp=100.0)

    store2 = EvidenceStore()
    store2.add_evidence(ev)
    snap2 = store2.create_decision_snapshot("d1", as_of_timestamp=100.0)

    assert snap1.id == snap2.id
    assert snap1.fingerprint() == snap2.fingerprint()


# ==============================================================================
# 9. Backward Compatibility
# ==============================================================================

def test_evidence_package_backward_compatibility():
    """Existing EvidenceCandidate and legacy Evidence contracts remain functional."""
    from researchforge.evidence import (
        EvidenceCandidate,
        EVIDENCE_CANDIDATE_SCHEMA,
        Evidence as LegacyEvidence,
        CanonicalEvidence,
        EvidenceNormalizer,
        EvidenceStore,
        EvidenceSnapshot,
    )

    cand = EvidenceCandidate(
        candidate_id="cand_test",
        source="openalex",
        relevance_score=0.91,
        retrieval_query="test query",
        retrieval_timestamp=100.0,
        retrieved_item={"title": "Test Paper"},
    )
    cand.validate()
    d = cand.to_dict()
    assert d["candidate_id"] == "cand_test"

    leg_ev = LegacyEvidence(
        evidence_id="leg_1",
        source="arxiv",
        title="Test",
        content="Content",
        relevance_score=0.88,
    )
    leg_ev.validate()
    assert leg_ev.evidence_id == "leg_1"
