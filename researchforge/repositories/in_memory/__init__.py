"""researchforge/repositories/in_memory package.

RF-1.0.0-alpha.3 (Phase 8A): In-memory repositories and Unit of Work.
"""
from .repos import (
    InMemoryArtifactRepository,
    InMemoryClaimRepository,
    InMemoryDecisionRepository,
    InMemoryDiagnosisRepository,
    InMemoryEntityRepository,
    InMemoryEvidenceRepository,
    InMemoryExperimentRunRepository,
    InMemoryExperimentSpecRepository,
    InMemoryFailureRepository,
    InMemoryHypothesisRepository,
    InMemoryOutcomeRepository,
    InMemoryProblemRepository,
    InMemoryProvenanceRepository,
    InMemoryQuestionRepository,
    InMemoryResearchStateRepository,
    InMemoryResearchSystemGenomeRepository,
    InMemoryTargetModelGenomeRepository,
)
from .uow import InMemoryUnitOfWork

__all__ = [
    "InMemoryArtifactRepository",
    "InMemoryClaimRepository",
    "InMemoryDecisionRepository",
    "InMemoryDiagnosisRepository",
    "InMemoryEntityRepository",
    "InMemoryEvidenceRepository",
    "InMemoryExperimentRunRepository",
    "InMemoryExperimentSpecRepository",
    "InMemoryFailureRepository",
    "InMemoryHypothesisRepository",
    "InMemoryOutcomeRepository",
    "InMemoryProblemRepository",
    "InMemoryProvenanceRepository",
    "InMemoryQuestionRepository",
    "InMemoryResearchStateRepository",
    "InMemoryResearchSystemGenomeRepository",
    "InMemoryTargetModelGenomeRepository",
    "InMemoryUnitOfWork",
]
