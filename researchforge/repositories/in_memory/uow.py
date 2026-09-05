"""researchforge/repositories/in_memory/uow.py — In-memory Unit of Work implementation.

RF-1.0.0-alpha.3 (Phase 8A): Transactional boundary supporting atomic commits and rollbacks in-memory.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from ..event_store import InMemoryEventStore
from ..interfaces import UnitOfWork
from ..vrdeg_adapter import InMemoryVRDEGRepository
from .repos import (
    InMemoryArtifactRepository,
    InMemoryClaimRepository,
    InMemoryDecisionRepository,
    InMemoryDiagnosisRepository,
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
    InMemoryActionRepository,
    InMemoryPolicyConfigRepository,
    InMemoryPortfolioBranchRepository,
    InMemorySaturationReportRepository,
    InMemoryGovernanceReviewRepository,
    InMemoryRetrospectiveRepository,
    InMemoryPolicyDecisionRecordRepository,
)


class InMemoryUnitOfWork:
    """In-memory UnitOfWork managing atomic transactional state across all repositories."""

    def __init__(self) -> None:
        self.problems = InMemoryProblemRepository()
        self.questions = InMemoryQuestionRepository()
        self.hypotheses = InMemoryHypothesisRepository()
        self.decisions = InMemoryDecisionRepository()
        self.target_genomes = InMemoryTargetModelGenomeRepository()
        self.system_genomes = InMemoryResearchSystemGenomeRepository()
        self.specs = InMemoryExperimentSpecRepository()
        self.runs = InMemoryExperimentRunRepository()
        self.outcomes = InMemoryOutcomeRepository()
        self.diagnoses = InMemoryDiagnosisRepository()
        self.failures = InMemoryFailureRepository()
        self.evidence = InMemoryEvidenceRepository()
        self.claims = InMemoryClaimRepository()
        self.states = InMemoryResearchStateRepository()
        self.provenance = InMemoryProvenanceRepository()
        self.artifacts = InMemoryArtifactRepository()
        self.graph = InMemoryVRDEGRepository()
        self.events = InMemoryEventStore()

        # Phase 9
        self.actions = InMemoryActionRepository()
        self.policy_configs = InMemoryPolicyConfigRepository()
        self.portfolio_branches = InMemoryPortfolioBranchRepository()
        self.saturation_reports = InMemorySaturationReportRepository()
        self.governance_reviews = InMemoryGovernanceReviewRepository()
        self.retrospectives = InMemoryRetrospectiveRepository()
        self.policy_decisions = InMemoryPolicyDecisionRecordRepository()

        self._in_transaction = False
        self._snapshot: Optional[Dict[str, Any]] = None

    def _all_repos(self) -> Dict[str, Any]:
        return {
            "problems": self.problems,
            "questions": self.questions,
            "hypotheses": self.hypotheses,
            "decisions": self.decisions,
            "target_genomes": self.target_genomes,
            "system_genomes": self.system_genomes,
            "specs": self.specs,
            "runs": self.runs,
            "outcomes": self.outcomes,
            "diagnoses": self.diagnoses,
            "failures": self.failures,
            "evidence": self.evidence,
            "claims": self.claims,
            "states": self.states,
            "provenance": self.provenance,
            "artifacts": self.artifacts,
            "actions": self.actions,
            "policy_configs": self.policy_configs,
            "portfolio_branches": self.portfolio_branches,
            "saturation_reports": self.saturation_reports,
            "governance_reviews": self.governance_reviews,
            "retrospectives": self.retrospectives,
            "policy_decisions": self.policy_decisions,
        }

    def _take_snapshot(self) -> Dict[str, Any]:
        snap = {}
        for name, repo in self._all_repos().items():
            snap[name] = copy.deepcopy(repo._items)
        snap["events_by_id"] = copy.deepcopy(self.events._events_by_id)
        snap["events_by_aggregate"] = copy.deepcopy(self.events._events_by_aggregate)
        snap["events_timeline"] = copy.deepcopy(self.events._timeline)
        snap["graph_nodes"] = copy.deepcopy(self.graph.graph._nodes)
        snap["graph_edges"] = copy.deepcopy(self.graph.graph._edges)
        return snap

    def _restore_snapshot(self, snap: Dict[str, Any]) -> None:
        for name, repo in self._all_repos().items():
            repo._items = copy.deepcopy(snap[name])
        self.events._events_by_id = copy.deepcopy(snap["events_by_id"])
        self.events._events_by_aggregate = copy.deepcopy(snap["events_by_aggregate"])
        self.events._timeline = copy.deepcopy(snap["events_timeline"])
        self.graph.graph._nodes = copy.deepcopy(snap["graph_nodes"])
        self.graph.graph._edges = copy.deepcopy(snap["graph_edges"])

    def __enter__(self) -> InMemoryUnitOfWork:
        self._snapshot = self._take_snapshot()
        self._in_transaction = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is not None:
                self.rollback()
        finally:
            self._in_transaction = False
            self._snapshot = None

    def commit(self) -> None:
        if not self._in_transaction:
            raise RuntimeError("Cannot commit outside an active transaction context.")
        # Changes are already in repositories; update transaction baseline
        self._snapshot = self._take_snapshot()

    def rollback(self) -> None:
        if self._snapshot is not None:
            self._restore_snapshot(self._snapshot)
