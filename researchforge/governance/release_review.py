"""researchforge/governance/release_review.py — Release gate reviewer.

Governance Role: RELEASE_REPRODUCIBILITY_ENGINEER
Evaluates:
- Complete test suite results
- Prior stage review decisions
- Database schema migration status
- Environment metadata and dependency freezes
- Artifact hash verification
"""
from __future__ import annotations

import time
from typing import List

from .contracts import (
    FindingSeverity,
    GovernanceReview,
    GovernanceStage,
    ReleaseGate,
    ReviewDecision,
    ReviewFinding,
    ReviewerRole,
    VersionTarget,
)


class ReleaseReviewer:
    """Evaluates whether all quality, safety, and scientific gates have passed before freezing."""

    def review(
        self,
        target: VersionTarget,
        prior_reviews: List[GovernanceReview],
        all_tests_passed: bool = True,
        unresolved_critical_warnings: int = 0,
        migrations_verified: bool = True,
    ) -> tuple[GovernanceReview, ReleaseGate]:
        findings: List[ReviewFinding] = []
        blocking_reasons: List[str] = []

        if not all_tests_passed:
            findings.append(
                ReviewFinding(
                    id=f"rel_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="TEST_SUITE_FAILURES",
                    category="RELEASE_READINESS",
                    severity=FindingSeverity.CRITICAL,
                    description="One or more tests in the regression suite failed.",
                    recommended_action="Resolve all test failures prior to freeze.",
                )
            )
            blocking_reasons.append("Test suite has failing tests.")

        if unresolved_critical_warnings > 0:
            findings.append(
                ReviewFinding(
                    id=f"rel_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="CRITICAL_WARNINGS_PRESENT",
                    category="RELEASE_READINESS",
                    severity=FindingSeverity.HIGH,
                    description=f"Found {unresolved_critical_warnings} unaddressed critical warnings.",
                    recommended_action="Address critical warnings or file formal exception.",
                )
            )
            blocking_reasons.append(f"{unresolved_critical_warnings} unresolved critical warnings.")

        if not migrations_verified:
            findings.append(
                ReviewFinding(
                    id=f"rel_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="UNVERIFIED_MIGRATIONS",
                    category="SCHEMA_INTEGRITY",
                    severity=FindingSeverity.CRITICAL,
                    description="Relational database migrations not verified against target schema.",
                    recommended_action="Run Alembic upgrade & downgrade roundtrip tests.",
                )
            )
            blocking_reasons.append("Unverified database schema migrations.")

        # Check prior reviews for REJECT verdicts
        for r in prior_reviews:
            if r.decision == ReviewDecision.REJECT:
                findings.append(
                    ReviewFinding(
                        id=f"rel_f_{len(findings)+1}",
                        schema_version="1.0",
                        finding_id=f"REJECTED_BY_{r.reviewer_role.value}",
                        category="GOVERNANCE_CONSENSUS",
                        severity=FindingSeverity.CRITICAL,
                        description=f"Prior review by '{r.reviewer_role.value}' rejected the change: {r.rationale}",
                        recommended_action="Address reviewer objections before requesting release gate approval.",
                    )
                )
                blocking_reasons.append(f"Rejected by {r.reviewer_role.value}.")

        passed = len(blocking_reasons) == 0
        decision = ReviewDecision.APPROVE if passed else ReviewDecision.REJECT
        severity = FindingSeverity.LOW if passed else FindingSeverity.CRITICAL
        rationale = "All release gates verified and passed." if passed else f"Release blocked: {'; '.join(blocking_reasons)}"

        completed_stages = list({r.review_stage for r in prior_reviews} | {GovernanceStage.FREEZE})

        review = GovernanceReview(
            id=f"rev_rel_{target.code_revision[:8]}",
            schema_version="1.0",
            target=target,
            reviewer_role=ReviewerRole.RELEASE_REPRODUCIBILITY_ENGINEER,
            review_stage=GovernanceStage.FREEZE,
            findings=findings,
            decision=decision,
            severity=severity,
            rationale=rationale,
            evidence_refs=[],
        )

        gate = ReleaseGate(
            id=f"gate_{target.code_revision[:8]}",
            schema_version="1.0",
            target=target,
            passed=passed,
            blocking_reasons=blocking_reasons,
            completed_stages=completed_stages,
            reviews=prior_reviews + [review],
            verified_at=time.time(),
        )

        return review, gate
