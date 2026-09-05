"""researchforge/repositories/in_memory/repos.py — In-memory repository implementations.

RF-1.0.0-alpha.3 (Phase 8A): Fast, zero-dependency in-memory repositories for local tests and execution.
"""
from __future__ import annotations

import copy
from typing import Dict, Generic, List, Optional, TypeVar

from ..interfaces import (
    ArtifactRepository,
    ClaimRepository,
    DecisionRepository,
    DiagnosisRepository,
    EntityImmutabilityError,
    EntityRepository,
    EvidenceRepository,
    ExperimentRunRepository,
    ExperimentSpecRepository,
    FailureRepository,
    HypothesisRepository,
    OutcomeRepository,
    ProblemRepository,
    ProvenanceRepository,
    QuestionRepository,
    ResearchStateRepository,
    ResearchSystemGenomeRepository,
    TargetModelGenomeRepository,
)
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

T = TypeVar("T")


class InMemoryEntityRepository(Generic[T]):
    """Generic in-memory repository storing deep-copied entities for isolation."""

    def __init__(self, is_immutable: bool = False) -> None:
        self._items: Dict[str, T] = {}
        self.is_immutable = is_immutable

    def get(self, id: str) -> Optional[T]:
        item = self._items.get(id)
        return copy.deepcopy(item) if item is not None else None

    def save(self, entity: T) -> None:
        entity_id = getattr(entity, "id", None) or getattr(entity, "artifact_id", None) or getattr(entity, "problem_id", None)
        if not entity_id:
            raise ValueError(f"Entity of type {type(entity).__name__} lacks an id.")
        if self.is_immutable and str(entity_id) in self._items:
            raise EntityImmutabilityError(f"Cannot overwrite immutable entity {type(entity).__name__} with id '{entity_id}'.")
        self._items[str(entity_id)] = copy.deepcopy(entity)

    def delete(self, id: str) -> bool:
        if self.is_immutable:
            raise EntityImmutabilityError(f"Cannot delete immutable historical record with id '{id}'.")
        if id in self._items:
            del self._items[id]
            return True
        return False

    def list_all(self) -> List[T]:
        return [copy.deepcopy(item) for item in self._items.values()]

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()


class InMemoryProblemRepository(InMemoryEntityRepository[ResearchProblem]):
    pass


class InMemoryQuestionRepository(InMemoryEntityRepository[ResearchQuestion]):
    pass


class InMemoryHypothesisRepository(InMemoryEntityRepository[Hypothesis]):
    pass


class InMemoryDecisionRepository(InMemoryEntityRepository[Decision]):
    pass


class InMemoryTargetModelGenomeRepository(InMemoryEntityRepository[TargetModelGenome]):
    pass


class InMemoryResearchSystemGenomeRepository(InMemoryEntityRepository[ResearchSystemGenome]):
    pass


class InMemoryExperimentSpecRepository(InMemoryEntityRepository[ExperimentSpec]):
    pass


class InMemoryExperimentRunRepository(InMemoryEntityRepository[ExperimentRun]):
    pass


class InMemoryOutcomeRepository(InMemoryEntityRepository[Outcome]):
    def __init__(self) -> None:
        super().__init__(is_immutable=True)


class InMemoryDiagnosisRepository(InMemoryEntityRepository[Diagnosis]):
    pass


class InMemoryFailureRepository(InMemoryEntityRepository[Failure]):
    def __init__(self) -> None:
        super().__init__(is_immutable=True)


class InMemoryEvidenceRepository(InMemoryEntityRepository[Evidence]):
    def get_by_claim(self, claim_id: str) -> List[Evidence]:
        return [
            copy.deepcopy(e) for e in self._items.values()
            if getattr(e, "claim_id", None) == claim_id
        ]


class InMemoryClaimRepository(InMemoryEntityRepository[Claim]):
    pass


class InMemoryResearchStateRepository(InMemoryEntityRepository[ResearchState]):
    def __init__(self) -> None:
        super().__init__(is_immutable=True)
        self._latest_id: Optional[str] = None

    def save(self, entity: ResearchState) -> None:
        super().save(entity)
        self._latest_id = entity.id

    def get_latest(self) -> Optional[ResearchState]:
        if self._latest_id is not None:
            return self.get(self._latest_id)
        all_states = self.list_all()
        return all_states[-1] if all_states else None


from ...domain.action import ResearchAction
from ...policy.config import PolicyConfig
from ...policy.portfolio import PortfolioBranch
from ...policy.saturation import SaturationReport
from ...policy.decision_record import PolicyDecisionRecord
from ...governance.contracts import GovernanceReview, RetrospectiveRecord

class InMemoryProvenanceRepository(InMemoryEntityRepository[Provenance]):
    def __init__(self) -> None:
        super().__init__(is_immutable=True)


class InMemoryArtifactRepository(InMemoryEntityRepository[Artifact]):
    def get_by_checksum(self, sha256: str) -> Optional[Artifact]:
        for art in self._items.values():
            if art.sha256.lower() == sha256.lower():
                return copy.deepcopy(art)
        return None


class InMemoryActionRepository(InMemoryEntityRepository[ResearchAction]):
    pass


class InMemoryPolicyConfigRepository(InMemoryEntityRepository[PolicyConfig]):
    def __init__(self) -> None:
        super().__init__(is_immutable=True)


class InMemoryPortfolioBranchRepository(InMemoryEntityRepository[PortfolioBranch]):
    pass


class InMemorySaturationReportRepository(InMemoryEntityRepository[SaturationReport]):
    def __init__(self) -> None:
        super().__init__(is_immutable=True)


class InMemoryGovernanceReviewRepository(InMemoryEntityRepository[GovernanceReview]):
    def __init__(self) -> None:
        super().__init__(is_immutable=True)


class InMemoryRetrospectiveRepository(InMemoryEntityRepository[RetrospectiveRecord]):
    def __init__(self) -> None:
        super().__init__(is_immutable=True)


class InMemoryPolicyDecisionRecordRepository(InMemoryEntityRepository[PolicyDecisionRecord]):
    def __init__(self) -> None:
        super().__init__(is_immutable=True)

