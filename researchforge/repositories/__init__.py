"""researchforge/repositories package.

RF-1.0.0-alpha.3 (Phase 8A): Professional persistence architecture.
Provides clean repository interfaces, in-memory reference implementations,
SQL (PostgreSQL / SQLite) backends, append-only event store, and Unit of Work.
"""
from .interfaces import (
    ArtifactRepository,
    ClaimRepository,
    DecisionRepository,
    DiagnosisRepository,
    EntityRepository,
    EventStoreRepository,
    EvidenceRepository,
    ExperimentRunRepository,
    ExperimentSpecRepository,
    FailureRepository,
    HypothesisRepository,
    OutcomeRepository,
    ProblemRepository,
    ProvenanceRepository,
    QuestionRepository,
    ResearchGraphRepository,
    ResearchStateRepository,
    ResearchSystemGenomeRepository,
    TargetModelGenomeRepository,
    UnitOfWork,
)
from .event_store import (
    EventOrderingError,
    EventStoreMutationError,
    InMemoryEventStore,
    ResearchEventRecord,
)
from .vrdeg_adapter import InMemoryVRDEGRepository
from .in_memory import InMemoryUnitOfWork
from .sql import SqlUnitOfWork, Base
from .replay import EventReplayService

__all__ = [
    "EntityRepository",
    "ProblemRepository",
    "QuestionRepository",
    "HypothesisRepository",
    "DecisionRepository",
    "TargetModelGenomeRepository",
    "ResearchSystemGenomeRepository",
    "ExperimentSpecRepository",
    "ExperimentRunRepository",
    "OutcomeRepository",
    "DiagnosisRepository",
    "FailureRepository",
    "EvidenceRepository",
    "ClaimRepository",
    "ResearchStateRepository",
    "ProvenanceRepository",
    "ArtifactRepository",
    "ResearchGraphRepository",
    "EventStoreRepository",
    "UnitOfWork",
    "ResearchEventRecord",
    "EventStoreMutationError",
    "EventOrderingError",
    "InMemoryEventStore",
    "InMemoryVRDEGRepository",
    "InMemoryUnitOfWork",
    "SqlUnitOfWork",
    "Base",
    "EventReplayService",
]
