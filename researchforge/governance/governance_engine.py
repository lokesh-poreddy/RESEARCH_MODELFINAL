"""researchforge/governance/governance_engine.py — Meta-Development Loop Coordinator.

Coordinates the 9-stage engineering workflow for ResearchForge itself:
PLAN -> CHALLENGE -> IMPLEMENT -> REVIEW -> TEST -> SCIENTIFIC_VALIDATE -> BENCHMARK -> FREEZE -> RETROSPECT

Crucial Invariant:
This engine governs changes to the ResearchForge platform.
It never executes or interferes with the Research Loop on target problem models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .adversarial_review import AdversarialReviewer
from .architecture_review import ArchitectureReviewer
from .benchmark_review import BenchmarkReviewer
from .contracts import (
    GovernanceReview,
    GovernanceStage,
    ReleaseGate,
    RetrospectiveRecord,
    ReviewDecision,
    ReviewerRole,
    VersionTarget,
)
from .release_review import ReleaseReviewer
from .retrospective import RetrospectiveRecorder
from .safety_review import SafetyExecutionReviewer
from .scientific_review import ScientificMethodReviewer


class GovernanceEngine:
    """Orchestrates structured governance reviews across the meta-development lifecycle."""

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root or Path.cwd()
        self.architecture_reviewer = ArchitectureReviewer(self.workspace_root)
        self.scientific_reviewer = ScientificMethodReviewer()
        self.safety_reviewer = SafetyExecutionReviewer()
        self.adversarial_reviewer = AdversarialReviewer()
        self.benchmark_reviewer = BenchmarkReviewer()
        self.release_reviewer = ReleaseReviewer()
        self.retrospective_recorder = RetrospectiveRecorder()
        self._reviews: List[GovernanceReview] = []

    def record_review(self, review: GovernanceReview) -> None:
        self._reviews.append(review)

    def list_reviews(self) -> List[GovernanceReview]:
        return list(self._reviews)

    def run_full_governance_cycle(
        self,
        target: VersionTarget,
        all_tests_passed: bool = True,
        unresolved_warnings: int = 0,
        migrations_verified: bool = True,
    ) -> ReleaseGate:
        """Executes the canonical review sequence up to the release freeze gate."""
        # 1. Architecture Review
        arch_rev = self.architecture_reviewer.review(target)
        self.record_review(arch_rev)

        # 2. Safety Review
        safe_rev = self.safety_reviewer.review(target)
        self.record_review(safe_rev)

        # 3. Scientific Review
        sci_rev = self.scientific_reviewer.review(target)
        self.record_review(sci_rev)

        # 4. Adversarial Review
        adv_rev = self.adversarial_reviewer.review(target)
        self.record_review(adv_rev)

        # 5. Benchmark Review
        bench_rev = self.benchmark_reviewer.review(target)
        self.record_review(bench_rev)

        # 6. Release Gate Review
        rel_rev, gate = self.release_reviewer.review(
            target=target,
            prior_reviews=list(self._reviews),
            all_tests_passed=all_tests_passed,
            unresolved_critical_warnings=unresolved_warnings,
            migrations_verified=migrations_verified,
        )
        self.record_review(rel_rev)

        return gate
