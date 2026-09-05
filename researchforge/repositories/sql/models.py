"""researchforge/repositories/sql/models.py — SQLAlchemy 2.x Declarative Models.

RF-1.0.0-alpha.3 (Phase 8A): Relational schema definitions corresponding 1:1 to canonical domain contracts.
Supports PostgreSQL (with JSONB when available) and SQLite (with JSON emulation).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    Float,
    Integer,
    String,
    Text,
    JSON,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ProblemModel(Base):
    __tablename__ = "research_problems"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    tags_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class QuestionModel(Base):
    __tablename__ = "research_questions"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    problem_id: Mapped[str] = mapped_column(String(128), index=True)
    question_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="OPEN")
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class HypothesisModel(Base):
    __tablename__ = "hypotheses"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    research_question_id: Mapped[str] = mapped_column(String(128), index=True)
    statement: Mapped[str] = mapped_column(Text)
    prediction: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    assumptions_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="PROPOSED")
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class DecisionModel(Base):
    __tablename__ = "decisions"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    hypothesis_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    research_state_fingerprint: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    rsg_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    selected_tmg_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    selected_operator: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    decision_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence_refs_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    memory_refs_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    decision_timestamp: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    policy_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class TargetModelGenomeModel(Base):
    __tablename__ = "target_model_genomes"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    model_type: Mapped[str] = mapped_column(String(64))
    hyperparameters_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    architecture_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    data_pipeline_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    parent_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class ResearchSystemGenomeModel(Base):
    __tablename__ = "research_system_genomes"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    generation: Mapped[int] = mapped_column(Integer, default=0)
    strategy_space_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    decision_policy_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    failure_policy_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    memory_config_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class ExperimentSpecModel(Base):
    __tablename__ = "experiment_specs"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    research_problem_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    research_question_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    hypothesis_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    decision_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    target_model_genome_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    research_system_genome_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    dataset_ref: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    preprocessing_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    intervention_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    baseline_config_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    seeds_json: Mapped[Optional[List[int]]] = mapped_column(JSON, nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


class ExperimentRunModel(Base):
    __tablename__ = "experiment_runs"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    experiment_spec_id: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    start_time: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    end_time: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    environment_fingerprint: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    code_revision: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    dataset_fingerprint: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    model_fingerprint: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    produced_artifacts_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    failure_info_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


class OutcomeModel(Base):
    __tablename__ = "outcomes"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    measured_metrics_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    baseline_metrics_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    improvement_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    statistical_summary_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    success: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    failure_category: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    validity_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    artifact_refs_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


class DiagnosisModel(Base):
    __tablename__ = "diagnoses"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    failure_category: Mapped[str] = mapped_column(String(64))
    root_cause_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    suggested_remedy: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


class FailureModel(Base):
    __tablename__ = "failures"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    experiment_run_id: Mapped[str] = mapped_column(String(128), index=True)
    category: Mapped[str] = mapped_column(String(64))
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recoverable: Mapped[bool] = mapped_column(Boolean, default=False)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


class EvidenceModel(Base):
    __tablename__ = "evidence_records"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    source: Mapped[str] = mapped_column(String(128))
    source_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    evidence_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    source_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claim_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    relation: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    quality_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    originating_vrdeg_node_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    originating_experiment_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    originating_run_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    originating_outcome_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    originating_trajectory_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class ClaimModel(Base):
    __tablename__ = "claims"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    statement: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="SPECULATIVE")
    hypothesis_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    problem_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    supporting_evidence_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    contradicting_evidence_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    uncertain_evidence_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    speculative_evidence_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class ResearchStateModel(Base):
    __tablename__ = "research_states"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    research_phase: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    problem_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    active_question_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    hypotheses_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    current_decision_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    selected_tmg_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    selected_rsg_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    recent_experiment_refs_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    recent_evidence_refs_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    recent_failures_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    memory_context_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    best_known_result_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    budget_consumed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    budget_remaining: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unresolved_contradictions_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    policy_state_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=time.time)


class ProvenanceModel(Base):
    __tablename__ = "provenance_records"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[str] = mapped_column(String(64))
    parents_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class ArtifactModel(Base):
    __tablename__ = "research_artifacts"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    artifact_type: Mapped[str] = mapped_column(String(64), index=True)
    uri_or_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    producer: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    code_revision: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    experiment_run_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_retired: Mapped[bool] = mapped_column(Boolean, default=False)


class EventModel(Base):
    __tablename__ = "research_events"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    aggregate_id: Mapped[str] = mapped_column(String(128), index=True)
    aggregate_type: Mapped[str] = mapped_column(String(64))
    sequence_number: Mapped[int] = mapped_column(Integer, index=True)
    global_sequence: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    payload_json: Mapped[Dict[str, Any]] = mapped_column(JSON)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    deterministic_fingerprint: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("aggregate_id", "sequence_number", name="uq_agg_seq"),
    )


class VRDEGNodeModel(Base):
    __tablename__ = "vrdeg_nodes"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    node_type: Mapped[str] = mapped_column(String(64), index=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    entity_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    parent_version_of: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class VRDEGEdgeModel(Base):
    __tablename__ = "vrdeg_edges"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(128), index=True)
    target_id: Mapped[str] = mapped_column(String(128), index=True)
    relation: Mapped[str] = mapped_column(String(64), index=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class ActionModel(Base):
    __tablename__ = "research_actions"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    action_type: Mapped[str] = mapped_column(String(64), index=True)
    target_context_json: Mapped[Dict[str, Any]] = mapped_column(JSON)
    expected_objective: Mapped[str] = mapped_column(Text)
    parameters_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    configuration_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class PolicyConfigModel(Base):
    __tablename__ = "policy_configs"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    policy_version: Mapped[str] = mapped_column(String(64), index=True)
    config_label: Mapped[str] = mapped_column(String(64))
    ablation_mode: Mapped[str] = mapped_column(String(64))
    w_performance: Mapped[float] = mapped_column(Float)
    w_information_gain: Mapped[float] = mapped_column(Float)
    w_novelty: Mapped[float] = mapped_column(Float)
    w_evidence: Mapped[float] = mapped_column(Float)
    w_transfer: Mapped[float] = mapped_column(Float)
    w_failure_risk: Mapped[float] = mapped_column(Float)
    w_redundancy: Mapped[float] = mapped_column(Float)
    w_cost: Mapped[float] = mapped_column(Float)
    min_confidence: Mapped[float] = mapped_column(Float, default=0.1)
    max_candidates: Mapped[int] = mapped_column(Integer, default=20)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class PortfolioBranchModel(Base):
    __tablename__ = "portfolio_branches"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    originating_hypothesis_id: Mapped[str] = mapped_column(String(128), index=True)
    action_json: Mapped[Dict[str, Any]] = mapped_column(JSON)
    score: Mapped[float] = mapped_column(Float)
    expected_value: Mapped[float] = mapped_column(Float)
    resource_allocation: Mapped[float] = mapped_column(Float)
    diversity_features_json: Mapped[Dict[str, str]] = mapped_column(JSON)
    branch_state: Mapped[str] = mapped_column(String(32), index=True)
    history_experiment_ids_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    failure_history_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class SaturationReportModel(Base):
    __tablename__ = "saturation_reports"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    current_status: Mapped[str] = mapped_column(String(64), index=True)
    recommendation: Mapped[str] = mapped_column(String(64))
    signal_values_json: Mapped[Dict[str, float]] = mapped_column(JSON)
    thresholds_json: Mapped[Dict[str, float]] = mapped_column(JSON)
    triggering_evidence_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    rationale: Mapped[str] = mapped_column(Text)
    evaluated_at: Mapped[float] = mapped_column(Float)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class GovernanceReviewModel(Base):
    __tablename__ = "governance_reviews"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    target_json: Mapped[Dict[str, Any]] = mapped_column(JSON)
    reviewer_role: Mapped[str] = mapped_column(String(64), index=True)
    review_stage: Mapped[str] = mapped_column(String(64), index=True)
    findings_json: Mapped[List[Dict[str, Any]]] = mapped_column(JSON)
    decision: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str] = mapped_column(String(32))
    rationale: Mapped[str] = mapped_column(Text)
    evidence_refs_json: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[float] = mapped_column(Float)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class RetrospectiveModel(Base):
    __tablename__ = "retrospectives"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    phase: Mapped[str] = mapped_column(String(64), index=True)
    target_json: Mapped[Dict[str, Any]] = mapped_column(JSON)
    what_changed_json: Mapped[List[str]] = mapped_column(JSON)
    what_passed_json: Mapped[List[str]] = mapped_column(JSON)
    what_failed_json: Mapped[List[str]] = mapped_column(JSON)
    unexpected_behavior_json: Mapped[List[str]] = mapped_column(JSON)
    negative_results_json: Mapped[List[str]] = mapped_column(JSON)
    architectural_debt_json: Mapped[List[str]] = mapped_column(JSON)
    research_insight: Mapped[str] = mapped_column(Text)
    benchmark_insight: Mapped[str] = mapped_column(Text)
    next_upgrade_hypothesis: Mapped[str] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column(Float)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class PolicyDecisionModel(Base):
    __tablename__ = "policy_decisions"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0")
    decision_id: Mapped[str] = mapped_column(String(128), index=True)
    research_state_fingerprint: Mapped[str] = mapped_column(String(64))
    evidence_snapshot_id: Mapped[str] = mapped_column(String(128))
    policy_version: Mapped[str] = mapped_column(String(64), index=True)
    policy_fingerprint: Mapped[str] = mapped_column(String(64))
    selected_action_id: Mapped[str] = mapped_column(String(128))
    selected_action_json: Mapped[Dict[str, Any]] = mapped_column(JSON)
    rejected_action_ids_json: Mapped[List[str]] = mapped_column(JSON)
    score_decompositions_json: Mapped[Dict[str, Any]] = mapped_column(JSON)
    policy_weights_json: Mapped[Dict[str, float]] = mapped_column(JSON)
    decision_timestamp: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str] = mapped_column(Text)
    available_memory_ref: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    credit_assignment_meta_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

