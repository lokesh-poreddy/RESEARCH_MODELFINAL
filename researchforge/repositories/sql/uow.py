"""researchforge/repositories/sql/uow.py — SQLAlchemy Unit of Work implementation.

RF-1.0.0-alpha.3 (Phase 8A): Relational transaction boundary wrapping SQLAlchemy sessions.
Guarantees atomic commit or rollback across complete research iterations.
"""
from __future__ import annotations

from typing import Any, Optional
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .repos import (
    SqlArtifactRepository,
    SqlClaimRepository,
    SqlDecisionRepository,
    SqlDiagnosisRepository,
    SqlEventStoreRepository,
    SqlEvidenceRepository,
    SqlExperimentRunRepository,
    SqlExperimentSpecRepository,
    SqlFailureRepository,
    SqlHypothesisRepository,
    SqlOutcomeRepository,
    SqlProblemRepository,
    SqlProvenanceRepository,
    SqlQuestionRepository,
    SqlResearchStateRepository,
    SqlResearchSystemGenomeRepository,
    SqlTargetModelGenomeRepository,
    SqlVRDEGRepository,
    SqlActionRepository,
    SqlPolicyConfigRepository,
    SqlPortfolioBranchRepository,
    SqlSaturationReportRepository,
    SqlGovernanceReviewRepository,
    SqlRetrospectiveRepository,
    SqlPolicyDecisionRepository,
)


class SqlUnitOfWork:
    """SQL-backed Unit of Work providing ACID transaction management."""

    def __init__(self, engine_or_factory: Engine | sessionmaker[Session]) -> None:
        if isinstance(engine_or_factory, Engine):
            self.session_factory = sessionmaker(bind=engine_or_factory, expire_on_commit=False)
        else:
            self.session_factory = engine_or_factory

        self.session: Optional[Session] = None
        self._in_transaction = False

    def __enter__(self) -> SqlUnitOfWork:
        self.session = self.session_factory()
        self._in_transaction = True

        # Wire repositories to active session
        self.problems = SqlProblemRepository(self.session)
        self.questions = SqlQuestionRepository(self.session)
        self.hypotheses = SqlHypothesisRepository(self.session)
        self.decisions = SqlDecisionRepository(self.session)
        self.target_genomes = SqlTargetModelGenomeRepository(self.session)
        self.system_genomes = SqlResearchSystemGenomeRepository(self.session)
        self.specs = SqlExperimentSpecRepository(self.session)
        self.runs = SqlExperimentRunRepository(self.session)
        self.outcomes = SqlOutcomeRepository(self.session)
        self.diagnoses = SqlDiagnosisRepository(self.session)
        self.failures = SqlFailureRepository(self.session)
        self.evidence = SqlEvidenceRepository(self.session)
        self.claims = SqlClaimRepository(self.session)
        self.states = SqlResearchStateRepository(self.session)
        self.provenance = SqlProvenanceRepository(self.session)
        self.artifacts = SqlArtifactRepository(self.session)
        self.graph = SqlVRDEGRepository(self.session)
        self.events = SqlEventStoreRepository(self.session)

        # Phase 9 additions
        self.actions = SqlActionRepository(self.session)
        self.policy_configs = SqlPolicyConfigRepository(self.session)
        self.portfolio_branches = SqlPortfolioBranchRepository(self.session)
        self.saturation_reports = SqlSaturationReportRepository(self.session)
        self.governance_reviews = SqlGovernanceReviewRepository(self.session)
        self.retrospectives = SqlRetrospectiveRepository(self.session)
        self.policy_decisions = SqlPolicyDecisionRepository(self.session)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is not None:
                self.rollback()
        finally:
            if self.session is not None:
                self.session.close()
                self.session = None
            self._in_transaction = False

    def commit(self) -> None:
        if not self._in_transaction or self.session is None:
            raise RuntimeError("Cannot commit outside an active transaction context.")
        self.session.commit()

    def rollback(self) -> None:
        if self.session is not None:
            self.session.rollback()
