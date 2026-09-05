"""researchforge/repositories/sql/repos.py — Concrete SQLAlchemy repository implementations.

RF-1.0.0-alpha.3 (Phase 8A): Maps domain contracts to relational models over an active SQLAlchemy Session.
Works seamlessly across PostgreSQL and SQLite dialects.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Generic, List, Optional, TypeVar
from sqlalchemy import select, delete, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import (
    ArtifactModel,
    ClaimModel,
    DecisionModel,
    DiagnosisModel,
    EventModel,
    EvidenceModel,
    ExperimentRunModel,
    ExperimentSpecModel,
    FailureModel,
    HypothesisModel,
    OutcomeModel,
    ProblemModel,
    ProvenanceModel,
    QuestionModel,
    ResearchStateModel,
    ResearchSystemGenomeModel,
    TargetModelGenomeModel,
    VRDEGEdgeModel,
    VRDEGNodeModel,
    ActionModel,
    PolicyConfigModel,
    PortfolioBranchModel,
    SaturationReportModel,
    GovernanceReviewModel,
    RetrospectiveModel,
    PolicyDecisionModel,
)
from ..interfaces import EntityImmutabilityError
from ..event_store import EventOrderingError, EventStoreMutationError, ResearchEventRecord
from ...domain.problem import ResearchProblem
from ...domain.question import ResearchQuestion
from ...domain.hypothesis import Hypothesis
from ...domain.decision import Decision
from ...domain.genome import TargetModelGenome, ResearchSystemGenome
from ...domain.experiment import ExperimentSpec, ExperimentRun
from ...domain.outcome import Outcome
from ...domain.diagnosis import Diagnosis
from ...domain.failure import Failure
from ...domain.evidence import Evidence
from ...domain.claim import Claim
from ...domain.state import ResearchState
from ...domain.provenance import Provenance
from ...artifacts.model import Artifact
from ...vrdeg.node import GraphNode, NodeType
from ...vrdeg.edge import GraphEdge, RelationType
from ...domain.action import ResearchAction
from ...policy.config import PolicyConfig
from ...policy.portfolio import PortfolioBranch
from ...policy.saturation import SaturationReport
from ...policy.decision_record import PolicyDecisionRecord
from ...governance.contracts import GovernanceReview, RetrospectiveRecord

T = TypeVar("T")
M = TypeVar("M")


class BaseSqlRepository(Generic[T, M]):
    """Base class providing CRUD operations over a SQLAlchemy session."""

    def __init__(self, session: Session, model_cls: type[M], is_immutable: bool = False) -> None:
        self.session = session
        self.model_cls = model_cls
        self.is_immutable = is_immutable

    def _to_domain(self, row: M) -> T:
        raise NotImplementedError

    def _from_domain(self, entity: T) -> M:
        raise NotImplementedError

    def get(self, id: str) -> Optional[T]:
        row = self.session.get(self.model_cls, id)
        return self._to_domain(row) if row is not None else None

    def save(self, entity: T) -> None:
        entity_id = getattr(entity, "id", None)
        existing = self.session.get(self.model_cls, entity_id) if entity_id else None
        if existing is not None and self.is_immutable:
            raise EntityImmutabilityError(f"Cannot overwrite immutable entity {type(entity).__name__} with id '{entity_id}'.")
        new_row = self._from_domain(entity)
        if existing is not None:
            self.session.merge(new_row)
        else:
            self.session.add(new_row)
        self.session.flush()

    def delete(self, id: str) -> bool:
        if self.is_immutable:
            raise EntityImmutabilityError(f"Cannot delete immutable historical record {self.model_cls.__name__} with id '{id}'.")
        row = self.session.get(self.model_cls, id)
        if row is not None:
            self.session.delete(row)
            self.session.flush()
            return True
        return False

    def list_all(self) -> List[T]:
        stmt = select(self.model_cls)
        rows = self.session.scalars(stmt).all()
        return [self._to_domain(r) for r in rows]

    def __len__(self) -> int:
        stmt = select(self.model_cls)
        return len(self.session.scalars(stmt).all())


class SqlProblemRepository(BaseSqlRepository[ResearchProblem, ProblemModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ProblemModel)

    def _to_domain(self, row: ProblemModel) -> ResearchProblem:
        return ResearchProblem(
            id=row.id,
            schema_version=row.schema_version,
            title=row.title,
            description=row.description,
            created_at=row.created_at,
            tags=row.tags_json,
            metadata=row.metadata_json,
        )

    def _from_domain(self, entity: ResearchProblem) -> ProblemModel:
        return ProblemModel(
            id=entity.id,
            schema_version=entity.schema_version,
            title=entity.title,
            description=entity.description,
            created_at=getattr(entity, "created_at", None),
            tags_json=getattr(entity, "tags", None),
            provenance_id=getattr(entity, "provenance_id", None),
            metadata_json=getattr(entity, "metadata", None),
        )


class SqlQuestionRepository(BaseSqlRepository[ResearchQuestion, QuestionModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, QuestionModel)

    def _to_domain(self, row: QuestionModel) -> ResearchQuestion:
        from researchforge.domain.question import QuestionStatus
        status_val = QuestionStatus(row.status) if row.status else QuestionStatus.OPEN
        return ResearchQuestion(
            id=row.id,
            schema_version=row.schema_version,
            problem_id=row.problem_id,
            question_text=row.question_text,
            status=status_val,
        )

    def _from_domain(self, entity: ResearchQuestion) -> QuestionModel:
        status_val = entity.status.value if hasattr(entity.status, "value") else str(entity.status)
        return QuestionModel(
            id=entity.id,
            schema_version=entity.schema_version,
            problem_id=entity.problem_id,
            question_text=entity.question_text,
            status=status_val,
            provenance_id=getattr(entity, "provenance_id", None),
            metadata_json=getattr(entity, "metadata", None),
        )


class SqlHypothesisRepository(BaseSqlRepository[Hypothesis, HypothesisModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, HypothesisModel)

    def _to_domain(self, row: HypothesisModel) -> Hypothesis:
        assumptions = tuple(row.assumptions_json) if row.assumptions_json else ()
        return Hypothesis(
            id=row.id,
            schema_version=row.schema_version,
            research_question_id=row.research_question_id,
            statement=row.statement,
            prediction=row.prediction,
            assumptions=assumptions,
            provenance_id=row.provenance_id,
        )

    def _from_domain(self, entity: Hypothesis) -> HypothesisModel:
        return HypothesisModel(
            id=entity.id,
            schema_version=entity.schema_version,
            research_question_id=entity.research_question_id,
            statement=entity.statement,
            prediction=entity.prediction,
            assumptions_json=list(entity.assumptions or ()),
            status=getattr(entity, "status", None),
            provenance_id=entity.provenance_id,
            metadata_json=getattr(entity, "metadata", None),
        )


class SqlDecisionRepository(BaseSqlRepository[Decision, DecisionModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, DecisionModel)

    def _to_domain(self, row: DecisionModel) -> Decision:
        return Decision(
            id=row.id,
            schema_version=row.schema_version,
            research_state_fingerprint=row.research_state_fingerprint,
            rsg_id=row.rsg_id,
            hypothesis_id=row.hypothesis_id,
            selected_tmg_id=row.selected_tmg_id,
            selected_operator=row.selected_operator,
            decision_reason=row.decision_reason,
            evidence_refs=row.evidence_refs_json,
            memory_refs=row.memory_refs_json,
            confidence=row.confidence,
            decision_timestamp=row.decision_timestamp,
            policy_version=row.policy_version,
        )

    def _from_domain(self, entity: Decision) -> DecisionModel:
        return DecisionModel(
            id=entity.id,
            schema_version=entity.schema_version,
            hypothesis_id=entity.hypothesis_id,
            research_state_fingerprint=getattr(entity, "research_state_fingerprint", None),
            rsg_id=getattr(entity, "rsg_id", None),
            selected_tmg_id=getattr(entity, "selected_tmg_id", None),
            selected_operator=getattr(entity, "selected_operator", None),
            decision_reason=getattr(entity, "decision_reason", None),
            evidence_refs_json=getattr(entity, "evidence_refs", None),
            memory_refs_json=getattr(entity, "memory_refs", None),
            confidence=getattr(entity, "confidence", None),
            decision_timestamp=getattr(entity, "decision_timestamp", None),
            policy_version=getattr(entity, "policy_version", None),
        )


class SqlTargetModelGenomeRepository(BaseSqlRepository[TargetModelGenome, TargetModelGenomeModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, TargetModelGenomeModel)

    def _to_domain(self, row: TargetModelGenomeModel) -> TargetModelGenome:
        return TargetModelGenome(
            id=row.id,
            schema_version=row.schema_version,
            model_type=row.model_type,
            hyperparameters=row.hyperparameters_json or {},
            architecture=row.architecture_json or {},
            data_pipeline=row.data_pipeline_json or {},
            parent_ids=row.parent_ids_json or [],
            generation=row.generation,
            created_at=row.created_at or 0.0,
        )

    def _from_domain(self, entity: TargetModelGenome) -> TargetModelGenomeModel:
        return TargetModelGenomeModel(
            id=entity.id,
            schema_version=entity.schema_version,
            model_type=entity.model_type,
            hyperparameters_json=entity.hyperparameters,
            architecture_json=entity.architecture,
            data_pipeline_json=entity.data_pipeline,
            parent_ids_json=entity.parent_ids,
            generation=entity.generation,
            created_at=entity.created_at,
        )


class SqlResearchSystemGenomeRepository(BaseSqlRepository[ResearchSystemGenome, ResearchSystemGenomeModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ResearchSystemGenomeModel)

    def _to_domain(self, row: ResearchSystemGenomeModel) -> ResearchSystemGenome:
        return ResearchSystemGenome(
            id=row.id,
            schema_version=row.schema_version,
            generation=row.generation,
            strategy_space=row.strategy_space_json or {},
            decision_policy=row.decision_policy_json or {},
            failure_policy=row.failure_policy_json or {},
            memory_config=row.memory_config_json or {},
            created_at=row.created_at or 0.0,
        )

    def _from_domain(self, entity: ResearchSystemGenome) -> ResearchSystemGenomeModel:
        return ResearchSystemGenomeModel(
            id=entity.id,
            schema_version=entity.schema_version,
            generation=entity.generation,
            strategy_space_json=entity.strategy_space,
            decision_policy_json=entity.decision_policy,
            failure_policy_json=entity.failure_policy,
            memory_config_json=entity.memory_config,
            created_at=entity.created_at,
        )


class SqlExperimentSpecRepository(BaseSqlRepository[ExperimentSpec, ExperimentSpecModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ExperimentSpecModel)

    def _to_domain(self, row: ExperimentSpecModel) -> ExperimentSpec:
        return ExperimentSpec(
            id=row.id,
            schema_version=row.schema_version,
            research_problem_id=row.research_problem_id,
            research_question_id=row.research_question_id,
            hypothesis_id=row.hypothesis_id,
            decision_id=row.decision_id,
            target_model_genome_id=row.target_model_genome_id,
            research_system_genome_id=row.research_system_genome_id,
            dataset_ref=row.dataset_ref,
            preprocessing=row.preprocessing_json,
            intervention_description=row.intervention_description,
            baseline_config=row.baseline_config_json,
            metrics=row.metrics_json,
            seeds=row.seeds_json,
            provenance=None,
        )

    def _from_domain(self, entity: ExperimentSpec) -> ExperimentSpecModel:
        return ExperimentSpecModel(
            id=entity.id,
            schema_version=entity.schema_version,
            research_problem_id=entity.research_problem_id,
            research_question_id=entity.research_question_id,
            hypothesis_id=entity.hypothesis_id,
            decision_id=entity.decision_id,
            target_model_genome_id=entity.target_model_genome_id or entity.tmg_id,
            research_system_genome_id=entity.research_system_genome_id or entity.rsg_id,
            dataset_ref=entity.dataset_ref or entity.dataset_id,
            preprocessing_json=entity.preprocessing,
            intervention_description=entity.intervention_description,
            baseline_config_json=entity.baseline_config,
            metrics_json=entity.metrics,
            seeds_json=entity.seeds,
            provenance_id=entity.provenance.id if entity.provenance else None,
        )


class SqlExperimentRunRepository(BaseSqlRepository[ExperimentRun, ExperimentRunModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ExperimentRunModel)

    def _to_domain(self, row: ExperimentRunModel) -> ExperimentRun:
        return ExperimentRun(
            id=row.id,
            schema_version=row.schema_version,
            experiment_spec_id=row.experiment_spec_id,
            spec_id=row.experiment_spec_id,
            start_time=row.start_time,
            end_time=row.end_time,
            status=row.status,
            environment_fingerprint=row.environment_fingerprint,
            code_revision=row.code_revision,
            dataset_fingerprint=row.dataset_fingerprint,
            model_fingerprint=row.model_fingerprint,
            seed=row.seed,
            produced_artifacts=row.produced_artifacts_json,
            failure_info=row.failure_info_json,
            provenance=None,
        )

    def _from_domain(self, entity: ExperimentRun) -> ExperimentRunModel:
        return ExperimentRunModel(
            id=entity.id,
            schema_version=entity.schema_version,
            experiment_spec_id=entity.experiment_spec_id or entity.spec_id,
            start_time=entity.start_time,
            end_time=entity.end_time,
            status=str(entity.status.value if hasattr(entity.status, "value") else entity.status),
            environment_fingerprint=entity.environment_fingerprint,
            code_revision=entity.code_revision,
            dataset_fingerprint=entity.dataset_fingerprint,
            model_fingerprint=entity.model_fingerprint,
            seed=entity.seed,
            produced_artifacts_json=entity.produced_artifacts,
            failure_info_json=entity.failure_info,
            provenance_id=entity.provenance.id if entity.provenance else None,
        )


class SqlOutcomeRepository(BaseSqlRepository[Outcome, OutcomeModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, OutcomeModel, is_immutable=True)

    def _to_domain(self, row: OutcomeModel) -> Outcome:
        d = {
            "id": row.id,
            "schema_version": row.schema_version,
            "run_id": row.run_id,
            "measured_metrics": row.measured_metrics_json,
            "baseline_metrics": row.baseline_metrics_json,
            "improvement": row.improvement_json,
            "statistical_summary": row.statistical_summary_json,
            "success": row.success,
            "failure_category": row.failure_category,
            "validity": row.validity_json,
            "artifact_refs": row.artifact_refs_json,
        }
        return Outcome.from_dict(d)

    def _from_domain(self, entity: Outcome) -> OutcomeModel:
        val_dict = entity.validity.to_dict() if entity.validity else None
        return OutcomeModel(
            id=entity.id,
            schema_version=entity.schema_version,
            run_id=entity.run_id,
            measured_metrics_json=entity.measured_metrics,
            baseline_metrics_json=entity.baseline_metrics,
            improvement_json=entity.improvement,
            statistical_summary_json=entity.statistical_summary,
            success=entity.success,
            failure_category=entity.failure_category,
            validity_json=val_dict,
            artifact_refs_json=entity.artifact_refs,
            provenance_id=None,
        )


class SqlDiagnosisRepository(BaseSqlRepository[Diagnosis, DiagnosisModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, DiagnosisModel)

    def _to_domain(self, row: DiagnosisModel) -> Diagnosis:
        return Diagnosis(
            id=row.id,
            schema_version=row.schema_version,
            run_id=row.run_id,
            failure_category=row.failure_category,
            root_cause_analysis=row.root_cause_analysis,
            suggested_remedy=row.suggested_remedy,
        )

    def _from_domain(self, entity: Diagnosis) -> DiagnosisModel:
        return DiagnosisModel(
            id=entity.id,
            schema_version=entity.schema_version,
            run_id=entity.run_id,
            failure_category=entity.failure_category,
            root_cause_analysis=entity.root_cause_analysis,
            suggested_remedy=entity.suggested_remedy,
            provenance_id=None,
        )


class SqlFailureRepository(BaseSqlRepository[Failure, FailureModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, FailureModel, is_immutable=True)

    def _to_domain(self, row: FailureModel) -> Failure:
        return Failure(
            id=row.id,
            schema_version=row.schema_version,
            experiment_id=row.experiment_run_id,
            failure_category=row.category,
        )

    def _from_domain(self, entity: Failure) -> FailureModel:
        return FailureModel(
            id=entity.id,
            schema_version=entity.schema_version,
            experiment_run_id=getattr(entity, "experiment_id", getattr(entity, "experiment_run_id", None)),
            category=getattr(entity, "failure_category", getattr(entity, "category", None)),
            error_message=getattr(entity, "error_message", None),
            recoverable=getattr(entity, "recoverable", False),
            provenance_id=getattr(entity, "provenance_id", None),
        )


class SqlEvidenceRepository(BaseSqlRepository[Evidence, EvidenceModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, EvidenceModel)

    def _to_domain(self, row: EvidenceModel) -> Evidence:
        d = {
            "id": row.id,
            "schema_version": row.schema_version,
            "source": row.source,
            "source_id": row.source_id,
            "evidence_type": row.evidence_type,
            "source_type": row.source_type,
            "claim_id": row.claim_id,
            "relation": row.relation,
            "provenance_id": row.provenance_id,
            "quality": row.quality_json,
            "confidence": row.confidence,
            "snippet": row.snippet,
            "originating_vrdeg_node_ids": row.originating_vrdeg_node_ids_json,
            "originating_experiment_ids": row.originating_experiment_ids_json,
            "originating_run_ids": row.originating_run_ids_json,
            "originating_outcome_ids": row.originating_outcome_ids_json,
            "originating_trajectory_ids": row.originating_trajectory_ids_json,
            "created_at": row.created_at,
            "metadata": row.metadata_json,
        }
        return Evidence.from_dict(d)

    def _from_domain(self, entity: Evidence) -> EvidenceModel:
        q_dict = entity.quality.to_dict() if entity.quality else None
        return EvidenceModel(
            id=entity.id,
            schema_version=entity.schema_version,
            source=entity.source,
            source_id=entity.source_id,
            evidence_type=entity.evidence_type,
            source_type=entity.source_type,
            claim_id=entity.claim_id,
            relation=entity.relation,
            provenance_id=entity.provenance_id,
            quality_json=q_dict,
            confidence=entity.confidence,
            snippet=entity.snippet,
            originating_vrdeg_node_ids_json=entity.originating_vrdeg_node_ids,
            originating_experiment_ids_json=entity.originating_experiment_ids,
            originating_run_ids_json=entity.originating_run_ids,
            originating_outcome_ids_json=entity.originating_outcome_ids,
            originating_trajectory_ids_json=entity.originating_trajectory_ids,
            created_at=entity.created_at,
            metadata_json=entity.metadata,
        )

    def get_by_claim(self, claim_id: str) -> List[Evidence]:
        stmt = select(EvidenceModel).where(EvidenceModel.claim_id == claim_id)
        rows = self.session.scalars(stmt).all()
        return [self._to_domain(r) for r in rows]


class SqlClaimRepository(BaseSqlRepository[Claim, ClaimModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ClaimModel)

    def _to_domain(self, row: ClaimModel) -> Claim:
        return Claim(
            id=row.id,
            schema_version=row.schema_version,
            statement=row.statement,
            claim_type=row.claim_type,
            confidence=row.confidence,
            status=row.status,
            hypothesis_id=row.hypothesis_id,
            problem_id=row.problem_id,
            supporting_evidence_ids=row.supporting_evidence_ids_json,
            contradicting_evidence_ids=row.contradicting_evidence_ids_json,
            uncertain_evidence_ids=row.uncertain_evidence_ids_json,
            speculative_evidence_ids=row.speculative_evidence_ids_json,
            provenance_id=row.provenance_id,
            created_at=row.created_at,
            metadata=row.metadata_json,
        )

    def _from_domain(self, entity: Claim) -> ClaimModel:
        return ClaimModel(
            id=entity.id,
            schema_version=entity.schema_version,
            statement=entity.statement,
            claim_type=entity.claim_type,
            confidence=entity.confidence,
            status=entity.status,
            hypothesis_id=entity.hypothesis_id,
            problem_id=entity.problem_id,
            supporting_evidence_ids_json=entity.supporting_evidence_ids,
            contradicting_evidence_ids_json=entity.contradicting_evidence_ids,
            uncertain_evidence_ids_json=entity.uncertain_evidence_ids,
            speculative_evidence_ids_json=entity.speculative_evidence_ids,
            provenance_id=entity.provenance_id,
            created_at=entity.created_at,
            metadata_json=entity.metadata,
        )


class SqlResearchStateRepository(BaseSqlRepository[ResearchState, ResearchStateModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ResearchStateModel, is_immutable=True)

    def _to_domain(self, row: ResearchStateModel) -> ResearchState:
        return ResearchState(
            id=row.id,
            schema_version=row.schema_version,
            problem_id=row.problem_id,
            active_question_id=row.active_question_id,
            hypotheses=row.hypotheses_json,
            current_decision_id=row.current_decision_id,
            selected_tmg_id=row.selected_tmg_id,
            selected_rsg_id=row.selected_rsg_id,
            recent_experiment_refs=row.recent_experiment_refs_json,
            recent_evidence_refs=row.recent_evidence_refs_json,
            recent_failures=row.recent_failures_json,
            memory_context=row.memory_context_json,
            best_known_result=row.best_known_result_json,
            budget_consumed=row.budget_consumed,
            budget_remaining=row.budget_remaining,
            research_phase=row.research_phase,
            unresolved_contradictions=row.unresolved_contradictions_json,
            policy_state=row.policy_state_json,
            provenance_id=row.provenance_id,
        )

    def _from_domain(self, entity: ResearchState) -> ResearchStateModel:
        return ResearchStateModel(
            id=entity.id,
            schema_version=entity.schema_version,
            problem_id=entity.problem_id,
            active_question_id=entity.active_question_id,
            hypotheses_json=entity.hypotheses,
            current_decision_id=entity.current_decision_id,
            selected_tmg_id=entity.selected_tmg_id,
            selected_rsg_id=entity.selected_rsg_id,
            recent_experiment_refs_json=entity.recent_experiment_refs,
            recent_evidence_refs_json=entity.recent_evidence_refs,
            recent_failures_json=entity.recent_failures,
            memory_context_json=entity.memory_context,
            best_known_result_json=entity.best_known_result,
            budget_consumed=entity.budget_consumed,
            budget_remaining=entity.budget_remaining,
            research_phase=entity.research_phase,
            unresolved_contradictions_json=entity.unresolved_contradictions,
            policy_state_json=entity.policy_state,
            provenance_id=entity.provenance_id,
        )

    def get_latest(self) -> Optional[ResearchState]:
        stmt = select(ResearchStateModel).order_by(ResearchStateModel.created_at.desc(), ResearchStateModel.id.desc()).limit(1)
        row = self.session.scalars(stmt).first()
        return self._to_domain(row) if row else None


class SqlProvenanceRepository(BaseSqlRepository[Provenance, ProvenanceModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ProvenanceModel, is_immutable=True)

    def _to_domain(self, row: ProvenanceModel) -> Provenance:
        return Provenance(
            id=row.id,
            schema_version=row.schema_version,
            created_by=row.created_by,
            created_at=row.created_at,
            parents=row.parents_json,
            notes=row.notes,
            metadata=row.metadata_json,
        )

    def _from_domain(self, entity: Provenance) -> ProvenanceModel:
        return ProvenanceModel(
            id=entity.id,
            schema_version=entity.schema_version,
            created_by=entity.created_by,
            created_at=entity.created_at,
            parents_json=entity.parents,
            notes=entity.notes,
            metadata_json=entity.metadata,
        )


class SqlArtifactRepository(BaseSqlRepository[Artifact, ArtifactModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ArtifactModel)

    def _to_domain(self, row: ArtifactModel) -> Artifact:
        return Artifact(
            id=row.id,
            schema_version=row.schema_version,
            artifact_type=row.artifact_type,
            uri_or_path=row.uri_or_path,
            sha256=row.sha256,
            size_bytes=row.size_bytes,
            producer=row.producer,
            code_revision=row.code_revision,
            experiment_run_id=row.experiment_run_id,
            provenance_id=row.provenance_id,
            mime_type=row.mime_type,
            metadata=row.metadata_json,
            created_at=row.created_at,
            is_retired=row.is_retired,
        )

    def _from_domain(self, entity: Artifact) -> ArtifactModel:
        return ArtifactModel(
            id=entity.id,
            schema_version=entity.schema_version,
            artifact_type=entity.artifact_type,
            uri_or_path=entity.uri_or_path,
            sha256=entity.sha256,
            size_bytes=entity.size_bytes,
            producer=entity.producer,
            code_revision=entity.code_revision,
            experiment_run_id=entity.experiment_run_id,
            provenance_id=entity.provenance_id,
            mime_type=entity.mime_type,
            metadata_json=entity.metadata,
            created_at=entity.created_at,
            is_retired=entity.is_retired,
        )

    def get_by_checksum(self, sha256: str) -> Optional[Artifact]:
        stmt = select(ArtifactModel).where(ArtifactModel.sha256 == sha256).limit(1)
        row = self.session.scalars(stmt).first()
        return self._to_domain(row) if row else None


class SqlEventStoreRepository:
    """Relational SQL append-only event store enforcing ordering and immutability."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def append(self, event: ResearchEventRecord) -> None:
        existing = self.session.get(EventModel, event.id)
        if existing is not None:
            raise EventStoreMutationError(f"Cannot overwrite historical event with id '{event.id}'.")

        # Check sequence ordering for aggregate
        stmt = (
            select(EventModel.sequence_number)
            .where(EventModel.aggregate_id == event.aggregate_id)
            .order_by(EventModel.sequence_number.desc())
            .limit(1)
        )
        last_seq = self.session.scalars(stmt).first()
        if last_seq is not None:
            if event.sequence_number <= last_seq:
                raise EventOrderingError(
                    f"Ordering violation for aggregate '{event.aggregate_id}': "
                    f"attempted to append sequence {event.sequence_number} after {last_seq}."
                )

        if event.global_sequence is not None:
            global_seq = event.global_sequence
        else:
            max_g = self.session.scalar(select(func.coalesce(func.max(EventModel.global_sequence), 0)))
            global_seq = (max_g or 0) + 1

        row = EventModel(
            id=event.id,
            schema_version=event.schema_version,
            event_type=event.event_type,
            aggregate_id=event.aggregate_id,
            aggregate_type=event.aggregate_type,
            sequence_number=event.sequence_number,
            global_sequence=global_seq,
            payload_json=event.payload,
            provenance_id=event.provenance_id,
            deterministic_fingerprint=event.fingerprint(),
            created_at=event.created_at,
        )
        self.session.add(row)
        try:
            self.session.flush()
        except IntegrityError as e:
            self.session.rollback()
            raise EventOrderingError(
                f"Concurrency conflict on aggregate '{event.aggregate_id}': "
                f"sequence {event.sequence_number} already exists."
            ) from e

    def get_by_aggregate(self, aggregate_id: str) -> List[ResearchEventRecord]:
        stmt = (
            select(EventModel)
            .where(EventModel.aggregate_id == aggregate_id)
            .order_by(EventModel.sequence_number.asc())
        )
        rows = self.session.scalars(stmt).all()
        return [
            ResearchEventRecord(
                id=r.id,
                schema_version=r.schema_version,
                event_type=r.event_type,
                aggregate_id=r.aggregate_id,
                aggregate_type=r.aggregate_type,
                sequence_number=r.sequence_number,
                global_sequence=r.global_sequence,
                payload=r.payload_json,
                provenance_id=r.provenance_id,
                created_at=r.created_at,
            )
            for r in rows
        ]

    def get_all_events(self, since_sequence: int = 0) -> List[ResearchEventRecord]:
        stmt = (
            select(EventModel)
            .where(EventModel.sequence_number >= since_sequence)
            .order_by(EventModel.global_sequence.asc(), EventModel.sequence_number.asc())
        )
        rows = self.session.scalars(stmt).all()
        return [
            ResearchEventRecord(
                id=r.id,
                schema_version=r.schema_version,
                event_type=r.event_type,
                aggregate_id=r.aggregate_id,
                aggregate_type=r.aggregate_type,
                sequence_number=r.sequence_number,
                global_sequence=r.global_sequence,
                payload=r.payload_json,
                provenance_id=r.provenance_id,
                created_at=r.created_at,
            )
            for r in rows
        ]

    def get_by_id(self, event_id: str) -> Optional[ResearchEventRecord]:
        r = self.session.get(EventModel, event_id)
        if r is None:
            return None
        return ResearchEventRecord(
            id=r.id,
            schema_version=r.schema_version,
            event_type=r.event_type,
            aggregate_id=r.aggregate_id,
            aggregate_type=r.aggregate_type,
            sequence_number=r.sequence_number,
            global_sequence=r.global_sequence,
            payload=r.payload_json,
            provenance_id=r.provenance_id,
            created_at=r.created_at,
        )


class SqlVRDEGRepository:
    """Relational SQL implementation of the ResearchGraphRepository for VRDEG."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_node(self, node: GraphNode) -> None:
        existing = self.session.get(VRDEGNodeModel, node.id)
        if existing is not None:
            raise ValueError(f"Duplicate node id in graph: '{node.id}'")
        row = VRDEGNodeModel(
            id=node.id,
            node_type=node.node_type if isinstance(node.node_type, str) else node.node_type.value,
            schema_version=node.schema_version,
            entity_id=getattr(node, "entity_id", getattr(node, "payload_ref", None)),
            parent_version_of=node.parent_version_of,
            provenance_id=node.provenance_id,
            created_at=node.created_at,
            payload_json=getattr(node, "payload", getattr(node, "metadata", None)),
        )
        self.session.add(row)
        self.session.flush()

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        r = self.session.get(VRDEGNodeModel, node_id)
        if r is None:
            return None
        return GraphNode(
            id=r.id,
            schema_version=r.schema_version,
            node_type=r.node_type,
            payload_ref=r.entity_id,
            parent_version_of=r.parent_version_of,
            provenance_id=r.provenance_id,
            created_at=r.created_at,
            metadata=r.payload_json,
        )

    def add_edge(self, edge: GraphEdge) -> None:
        existing = self.session.get(VRDEGEdgeModel, edge.id)
        if existing is not None:
            raise ValueError(f"Duplicate edge id in graph: '{edge.id}'")
        if not self.get_node(edge.source_id) or not self.get_node(edge.target_id):
            raise ValueError("Edge references non-existent node.")
        row = VRDEGEdgeModel(
            id=edge.id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation=edge.relation if isinstance(edge.relation, str) else edge.relation.value,
            schema_version=edge.schema_version,
            provenance_id=edge.provenance_id,
            created_at=edge.created_at,
            metadata_json=edge.metadata,
        )
        self.session.add(row)
        self.session.flush()

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        r = self.session.get(VRDEGEdgeModel, edge_id)
        if r is None:
            return None
        return GraphEdge(
            id=r.id,
            schema_version=r.schema_version,
            source_id=r.source_id,
            target_id=r.target_id,
            relation=RelationType(r.relation),
            provenance_id=r.provenance_id,
            created_at=r.created_at,
            metadata=r.metadata_json,
        )

    def get_edges_for_node(self, node_id: str) -> List[GraphEdge]:
        stmt = select(VRDEGEdgeModel).where(
            (VRDEGEdgeModel.source_id == node_id) | (VRDEGEdgeModel.target_id == node_id)
        )
        rows = self.session.scalars(stmt).all()
        return [
            GraphEdge(
                id=r.id,
                schema_version=r.schema_version,
                source_id=r.source_id,
                target_id=r.target_id,
                relation=RelationType(r.relation),
                provenance_id=r.provenance_id,
                created_at=r.created_at,
                metadata=r.metadata_json,
            )
            for r in rows
        ]

    def trace_lineage(self, start_node_id: str) -> List[GraphNode]:
        visited = set()
        lineage = []
        queue = [start_node_id]

        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)

            node = self.get_node(curr_id)
            if node:
                lineage.append(node)

            edges = self.get_edges_for_node(curr_id)
            for edge in edges:
                if edge.target_id == curr_id and edge.source_id not in visited:
                    queue.append(edge.source_id)

        return lineage

    def trace_branch(self, branch_id: str) -> List[GraphNode]:
        stmt = select(VRDEGEdgeModel).where(
            VRDEGEdgeModel.relation == RelationType.BRANCH_OF.value,
            VRDEGEdgeModel.target_id == branch_id,
        )
        edges = self.session.scalars(stmt).all()
        nodes = []
        for e in edges:
            node = self.get_node(e.source_id)
            if node:
                nodes.append(node)
        return nodes


class SqlActionRepository(BaseSqlRepository[ResearchAction, ActionModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, ActionModel)

    def _to_domain(self, row: ActionModel) -> ResearchAction:
        return ResearchAction(
            id=row.id,
            schema_version=row.schema_version,
            action_type=row.action_type,
            target_context=row.target_context_json,
            expected_objective=row.expected_objective,
            parameters=row.parameters_json,
            provenance_id=row.provenance_id,
            configuration_fingerprint=row.configuration_fingerprint,
            metadata=row.metadata_json,
        )

    def _from_domain(self, entity: ResearchAction) -> ActionModel:
        return ActionModel(
            id=entity.id,
            schema_version=entity.schema_version,
            action_type=entity.action_type.value if hasattr(entity.action_type, "value") else str(entity.action_type),
            target_context_json=entity.target_context,
            expected_objective=entity.expected_objective,
            parameters_json=entity.parameters,
            provenance_id=entity.provenance_id,
            configuration_fingerprint=entity.configuration_fingerprint,
            metadata_json=entity.metadata,
        )


class SqlPolicyConfigRepository(BaseSqlRepository[PolicyConfig, PolicyConfigModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, PolicyConfigModel, is_immutable=True)

    def _to_domain(self, row: PolicyConfigModel) -> PolicyConfig:
        from ...policy.config import PolicyAblationMode
        return PolicyConfig(
            id=row.id,
            schema_version=row.schema_version,
            policy_version=row.policy_version,
            config_label=row.config_label,
            ablation_mode=PolicyAblationMode(row.ablation_mode),
            w_performance=row.w_performance,
            w_information_gain=row.w_information_gain,
            w_novelty=row.w_novelty,
            w_evidence=row.w_evidence,
            w_transfer=row.w_transfer,
            w_failure_risk=row.w_failure_risk,
            w_redundancy=row.w_redundancy,
            w_cost=row.w_cost,
            min_confidence=row.min_confidence,
            max_candidates=row.max_candidates,
            provenance_id=row.provenance_id,
            metadata=row.metadata_json or {},
        )

    def _from_domain(self, entity: PolicyConfig) -> PolicyConfigModel:
        return PolicyConfigModel(
            id=entity.id,
            schema_version=entity.schema_version,
            policy_version=entity.policy_version,
            config_label=entity.config_label,
            ablation_mode=entity.ablation_mode.value if hasattr(entity.ablation_mode, "value") else str(entity.ablation_mode),
            w_performance=entity.w_performance,
            w_information_gain=entity.w_information_gain,
            w_novelty=entity.w_novelty,
            w_evidence=entity.w_evidence,
            w_transfer=entity.w_transfer,
            w_failure_risk=entity.w_failure_risk,
            w_redundancy=entity.w_redundancy,
            w_cost=entity.w_cost,
            min_confidence=entity.min_confidence,
            max_candidates=entity.max_candidates,
            provenance_id=entity.provenance_id,
            metadata_json=entity.metadata,
        )


class SqlPortfolioBranchRepository(BaseSqlRepository[PortfolioBranch, PortfolioBranchModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, PortfolioBranchModel)

    def _to_domain(self, row: PortfolioBranchModel) -> PortfolioBranch:
        from ...policy.portfolio import BranchState
        return PortfolioBranch(
            id=row.id,
            schema_version=row.schema_version,
            originating_hypothesis_id=row.originating_hypothesis_id,
            action=ResearchAction.from_dict(row.action_json),
            score=row.score,
            expected_value=row.expected_value,
            resource_allocation=row.resource_allocation,
            diversity_features=row.diversity_features_json,
            branch_state=BranchState(row.branch_state),
            history_experiment_ids=row.history_experiment_ids_json or [],
            failure_history=row.failure_history_json or [],
            provenance_id=row.provenance_id,
            metadata=row.metadata_json or {},
        )

    def _from_domain(self, entity: PortfolioBranch) -> PortfolioBranchModel:
        return PortfolioBranchModel(
            id=entity.id,
            schema_version=entity.schema_version,
            originating_hypothesis_id=entity.originating_hypothesis_id,
            action_json=entity.action.to_dict(),
            score=entity.score,
            expected_value=entity.expected_value,
            resource_allocation=entity.resource_allocation,
            diversity_features_json=entity.diversity_features,
            branch_state=entity.branch_state.value if hasattr(entity.branch_state, "value") else str(entity.branch_state),
            history_experiment_ids_json=entity.history_experiment_ids,
            failure_history_json=entity.failure_history,
            provenance_id=entity.provenance_id,
            metadata_json=entity.metadata,
        )


class SqlSaturationReportRepository(BaseSqlRepository[SaturationReport, SaturationReportModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, SaturationReportModel, is_immutable=True)

    def _to_domain(self, row: SaturationReportModel) -> SaturationReport:
        from ...policy.saturation import SaturationState, PivotRecommendation
        return SaturationReport(
            id=row.id,
            schema_version=row.schema_version,
            current_status=SaturationState(row.current_status),
            recommendation=PivotRecommendation(row.recommendation),
            signal_values=row.signal_values_json,
            thresholds=row.thresholds_json,
            triggering_evidence=row.triggering_evidence_json or [],
            rationale=row.rationale,
            evaluated_at=row.evaluated_at,
            provenance_id=row.provenance_id,
            metadata=row.metadata_json or {},
        )

    def _from_domain(self, entity: SaturationReport) -> SaturationReportModel:
        return SaturationReportModel(
            id=entity.id,
            schema_version=entity.schema_version,
            current_status=entity.current_status.value if hasattr(entity.current_status, "value") else str(entity.current_status),
            recommendation=entity.recommendation.value if hasattr(entity.recommendation, "value") else str(entity.recommendation),
            signal_values_json=entity.signal_values,
            thresholds_json=entity.thresholds,
            triggering_evidence_json=entity.triggering_evidence,
            rationale=entity.rationale,
            evaluated_at=entity.evaluated_at,
            provenance_id=entity.provenance_id,
            metadata_json=entity.metadata,
        )


class SqlGovernanceReviewRepository(BaseSqlRepository[GovernanceReview, GovernanceReviewModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, GovernanceReviewModel, is_immutable=True)

    def _to_domain(self, row: GovernanceReviewModel) -> GovernanceReview:
        from ...governance.contracts import VersionTarget, ReviewerRole, GovernanceStage, FindingSeverity, ReviewDecision, ReviewFinding
        return GovernanceReview(
            id=row.id,
            schema_version=row.schema_version,
            target=VersionTarget.from_dict(row.target_json),
            reviewer_role=ReviewerRole(row.reviewer_role),
            review_stage=GovernanceStage(row.review_stage),
            findings=[ReviewFinding.from_dict(f) for f in row.findings_json],
            decision=ReviewDecision(row.decision),
            severity=FindingSeverity(row.severity),
            rationale=row.rationale,
            evidence_refs=row.evidence_refs_json or [],
            created_at=row.created_at,
            provenance_id=row.provenance_id,
            metadata=row.metadata_json or {},
        )

    def _from_domain(self, entity: GovernanceReview) -> GovernanceReviewModel:
        return GovernanceReviewModel(
            id=entity.id,
            schema_version=entity.schema_version,
            target_json=entity.target.to_dict(),
            reviewer_role=entity.reviewer_role.value if hasattr(entity.reviewer_role, "value") else str(entity.reviewer_role),
            review_stage=entity.review_stage.value if hasattr(entity.review_stage, "value") else str(entity.review_stage),
            findings_json=[f.to_dict() for f in entity.findings],
            decision=entity.decision.value if hasattr(entity.decision, "value") else str(entity.decision),
            severity=entity.severity.value if hasattr(entity.severity, "value") else str(entity.severity),
            rationale=entity.rationale,
            evidence_refs_json=entity.evidence_refs,
            created_at=entity.created_at,
            provenance_id=entity.provenance_id,
            metadata_json=entity.metadata,
        )


class SqlRetrospectiveRepository(BaseSqlRepository[RetrospectiveRecord, RetrospectiveModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, RetrospectiveModel, is_immutable=True)

    def _to_domain(self, row: RetrospectiveModel) -> RetrospectiveRecord:
        from ...governance.contracts import VersionTarget
        return RetrospectiveRecord(
            id=row.id,
            schema_version=row.schema_version,
            phase=row.phase,
            target=VersionTarget.from_dict(row.target_json),
            what_changed=row.what_changed_json or [],
            what_passed=row.what_passed_json or [],
            what_failed=row.what_failed_json or [],
            unexpected_behavior=row.unexpected_behavior_json or [],
            negative_results=row.negative_results_json or [],
            architectural_debt=row.architectural_debt_json or [],
            research_insight=row.research_insight,
            benchmark_insight=row.benchmark_insight,
            next_upgrade_hypothesis=row.next_upgrade_hypothesis,
            created_at=row.created_at,
            provenance_id=row.provenance_id,
            metadata=row.metadata_json or {},
        )

    def _from_domain(self, entity: RetrospectiveRecord) -> RetrospectiveModel:
        return RetrospectiveModel(
            id=entity.id,
            schema_version=entity.schema_version,
            phase=entity.phase,
            target_json=entity.target.to_dict(),
            what_changed_json=entity.what_changed,
            what_passed_json=entity.what_passed,
            what_failed_json=entity.what_failed,
            unexpected_behavior_json=entity.unexpected_behavior,
            negative_results_json=entity.negative_results,
            architectural_debt_json=entity.architectural_debt,
            research_insight=entity.research_insight,
            benchmark_insight=entity.benchmark_insight,
            next_upgrade_hypothesis=entity.next_upgrade_hypothesis,
            created_at=entity.created_at,
            provenance_id=entity.provenance_id,
            metadata_json=entity.metadata,
        )


class SqlPolicyDecisionRepository(BaseSqlRepository[PolicyDecisionRecord, PolicyDecisionModel]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, PolicyDecisionModel, is_immutable=True)

    def _to_domain(self, row: PolicyDecisionModel) -> PolicyDecisionRecord:
        from ...policy.score_decomposition import ScoreDecomposition
        return PolicyDecisionRecord(
            id=row.id,
            schema_version=row.schema_version,
            decision_id=row.decision_id,
            research_state_fingerprint=row.research_state_fingerprint,
            evidence_snapshot_id=row.evidence_snapshot_id,
            policy_version=row.policy_version,
            policy_fingerprint=row.policy_fingerprint,
            selected_action_id=row.selected_action_id,
            selected_action=ResearchAction.from_dict(row.selected_action_json),
            rejected_action_ids=row.rejected_action_ids_json or [],
            score_decompositions={
                k: ScoreDecomposition.from_dict(v) for k, v in row.score_decompositions_json.items()
            },
            policy_weights=row.policy_weights_json or {},
            decision_timestamp=row.decision_timestamp,
            explanation=row.explanation,
            available_memory_ref=row.available_memory_ref,
            provenance_id=row.provenance_id,
            credit_assignment_meta=row.credit_assignment_meta_json,
            metadata=row.metadata_json or {},
        )

    def _from_domain(self, entity: PolicyDecisionRecord) -> PolicyDecisionModel:
        return PolicyDecisionModel(
            id=entity.id,
            schema_version=entity.schema_version,
            decision_id=entity.decision_id,
            research_state_fingerprint=entity.research_state_fingerprint,
            evidence_snapshot_id=entity.evidence_snapshot_id,
            policy_version=entity.policy_version,
            policy_fingerprint=entity.policy_fingerprint,
            selected_action_id=entity.selected_action_id,
            selected_action_json=entity.selected_action.to_dict(),
            rejected_action_ids_json=entity.rejected_action_ids,
            score_decompositions_json={k: v.to_dict() for k, v in entity.score_decompositions.items()},
            policy_weights_json=entity.policy_weights,
            decision_timestamp=entity.decision_timestamp,
            explanation=entity.explanation,
            available_memory_ref=entity.available_memory_ref,
            provenance_id=entity.provenance_id,
            credit_assignment_meta_json=entity.credit_assignment_meta,
            metadata_json=entity.metadata,
        )
