"""researchforge/repositories/interfaces.py — Canonical Repository & Unit of Work Protocols.

RF-1.0.0-alpha.3 (Phase 8A): Database-agnostic interfaces decoupling domain models from storage engines.
"""
from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar

from ..domain.problem import ResearchProblem
from ..domain.question import ResearchQuestion
from ..domain.hypothesis import Hypothesis
from ..domain.decision import Decision
from ..domain.genome import TargetModelGenome, ResearchSystemGenome
from ..domain.experiment import ExperimentSpec, ExperimentRun
from ..domain.outcome import Outcome
from ..domain.diagnosis import Diagnosis
from ..domain.failure import Failure
from ..domain.evidence import Evidence
from ..domain.claim import Claim
from ..domain.state import ResearchState
from ..domain.provenance import Provenance
from ..artifacts.model import Artifact
from ..vrdeg.node import GraphNode
from ..vrdeg.edge import GraphEdge

from ..domain.action import ResearchAction
from ..policy.config import PolicyConfig
from ..policy.portfolio import PortfolioBranch
from ..policy.saturation import SaturationReport
from ..policy.decision_record import PolicyDecisionRecord
from ..governance.contracts import GovernanceReview, RetrospectiveRecord

T = TypeVar("T")


class EntityImmutabilityError(Exception):
    """Raised when an attempt is made to overwrite or delete an immutable historical research record."""
    pass


class EntityRepository(Protocol[T]):
    """Generic repository protocol for identity-bearing domain entities."""

    def get(self, id: str) -> Optional[T]:
        ...

    def save(self, entity: T) -> None:
        ...

    def delete(self, id: str) -> bool:
        ...

    def list_all(self) -> List[T]:
        ...


class ProblemRepository(EntityRepository[ResearchProblem], Protocol):
    pass


class QuestionRepository(EntityRepository[ResearchQuestion], Protocol):
    pass


class HypothesisRepository(EntityRepository[Hypothesis], Protocol):
    pass


class DecisionRepository(EntityRepository[Decision], Protocol):
    pass


class TargetModelGenomeRepository(EntityRepository[TargetModelGenome], Protocol):
    pass


class ResearchSystemGenomeRepository(EntityRepository[ResearchSystemGenome], Protocol):
    pass


class ExperimentSpecRepository(EntityRepository[ExperimentSpec], Protocol):
    pass


class ExperimentRunRepository(EntityRepository[ExperimentRun], Protocol):
    pass


class OutcomeRepository(EntityRepository[Outcome], Protocol):
    pass


class DiagnosisRepository(EntityRepository[Diagnosis], Protocol):
    pass


class FailureRepository(EntityRepository[Failure], Protocol):
    pass


class EvidenceRepository(EntityRepository[Evidence], Protocol):
    def get_by_claim(self, claim_id: str) -> List[Evidence]:
        ...


class ClaimRepository(EntityRepository[Claim], Protocol):
    pass


class ResearchStateRepository(EntityRepository[ResearchState], Protocol):
    def get_latest(self) -> Optional[ResearchState]:
        ...


class ProvenanceRepository(EntityRepository[Provenance], Protocol):
    pass


class ArtifactRepository(EntityRepository[Artifact], Protocol):
    def get_by_checksum(self, sha256: str) -> Optional[Artifact]:
        ...


class ActionRepository(EntityRepository[ResearchAction], Protocol):
    pass


class PolicyConfigRepository(EntityRepository[PolicyConfig], Protocol):
    pass


class PortfolioBranchRepository(EntityRepository[PortfolioBranch], Protocol):
    pass


class SaturationReportRepository(EntityRepository[SaturationReport], Protocol):
    pass


class GovernanceReviewRepository(EntityRepository[GovernanceReview], Protocol):
    pass


class RetrospectiveRepository(EntityRepository[RetrospectiveRecord], Protocol):
    pass


class PolicyDecisionRecordRepository(EntityRepository[PolicyDecisionRecord], Protocol):
    pass


class ResearchGraphRepository(Protocol):
    """Abstract protocol for VRDEG graph persistence."""

    def add_node(self, node: GraphNode) -> None:
        ...

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        ...

    def add_edge(self, edge: GraphEdge) -> None:
        ...

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        ...

    def get_edges_for_node(self, node_id: str) -> List[GraphEdge]:
        ...

    def trace_lineage(self, start_node_id: str) -> List[GraphNode]:
        ...

    def trace_branch(self, branch_id: str) -> List[GraphNode]:
        ...


class StoredEvent(Protocol):
    """Protocol for persisted append-only research events."""
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    sequence_number: int
    schema_version: str
    payload: Dict[str, Any]
    provenance_id: Optional[str]
    deterministic_fingerprint: str
    created_at: float


class EventStoreRepository(Protocol):
    """Append-only event journal protocol."""

    def append(self, event: Any) -> None:
        """Append an event. Raises EventStoreMutationError if ID exists or ordering is invalid."""
        ...

    def get_by_aggregate(self, aggregate_id: str) -> List[Any]:
        """Return all events for an aggregate in strictly ascending sequence order."""
        ...

    def get_all_events(self, since_sequence: int = 0) -> List[Any]:
        """Return all events across all aggregates in chronological / sequence order."""
        ...

    def get_by_id(self, event_id: str) -> Optional[Any]:
        ...


class UnitOfWork(Protocol):
    """Atomic transaction boundary across all research repositories."""

    problems: ProblemRepository
    questions: QuestionRepository
    hypotheses: HypothesisRepository
    decisions: DecisionRepository
    target_genomes: TargetModelGenomeRepository
    system_genomes: ResearchSystemGenomeRepository
    specs: ExperimentSpecRepository
    runs: ExperimentRunRepository
    outcomes: OutcomeRepository
    diagnoses: DiagnosisRepository
    failures: FailureRepository
    evidence: EvidenceRepository
    claims: ClaimRepository
    states: ResearchStateRepository
    provenance: ProvenanceRepository
    artifacts: ArtifactRepository
    graph: ResearchGraphRepository
    events: EventStoreRepository

    # Phase 9 additions
    actions: ActionRepository
    policy_configs: PolicyConfigRepository
    portfolio_branches: PortfolioBranchRepository
    saturation_reports: SaturationReportRepository
    governance_reviews: GovernanceReviewRepository
    retrospectives: RetrospectiveRepository
    policy_decisions: PolicyDecisionRecordRepository

    def __enter__(self) -> UnitOfWork:
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...

    def commit(self) -> None:
        """Persist all pending repository changes atomically."""
        ...

    def rollback(self) -> None:
        """Discard all pending changes in the transaction."""
        ...

