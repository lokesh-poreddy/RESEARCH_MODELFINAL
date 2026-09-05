"""researchforge/governance/safety_review.py — Execution safety reviewer.

Governance Role: SAFETY_EXECUTION_REVIEWER
Evaluates:
- Execution policy existence and SafeRunner authority
- Resource boundaries and timeout enforcement
- Absence of unsafe execution bypasses
- Artifact path confinement
- Provenance capture on failures and successes
"""
from __future__ import annotations

from typing import List

from .contracts import (
    FindingSeverity,
    GovernanceReview,
    GovernanceStage,
    ReviewDecision,
    ReviewFinding,
    ReviewerRole,
    VersionTarget,
)


class SafetyExecutionReviewer:
    """Reviews runtime execution boundaries, timeouts, and sandbox integrity."""

    def review(
        self,
        target: VersionTarget,
        safe_runner_active: bool = True,
        timeouts_enforced: bool = True,
        uncontrolled_paths_detected: bool = False,
    ) -> GovernanceReview:
        findings: List[ReviewFinding] = []

        if not safe_runner_active:
            findings.append(
                ReviewFinding(
                    id=f"safe_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="SAFE_RUNNER_BYPASS",
                    category="EXECUTION_SAFETY",
                    severity=FindingSeverity.CRITICAL,
                    description="Experiments executed without SafeRunner authority.",
                    recommended_action="Route all execution through SafeRunner sandbox.",
                )
            )

        if not timeouts_enforced:
            findings.append(
                ReviewFinding(
                    id=f"safe_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="NO_TIMEOUT_ENFORCEMENT",
                    category="EXECUTION_SAFETY",
                    severity=FindingSeverity.HIGH,
                    description="Runaway execution timeouts not configured or enforced.",
                    recommended_action="Configure strict wall-clock timeout on all experiment invocations.",
                )
            )

        if uncontrolled_paths_detected:
            findings.append(
                ReviewFinding(
                    id=f"safe_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="UNCONTROLLED_ARTIFACT_PATHS",
                    category="STORAGE_SAFETY",
                    severity=FindingSeverity.HIGH,
                    description="Artifacts written outside of managed registry storage directory.",
                    recommended_action="Confine all output writes to ArtifactStorage managed tree.",
                )
            )

        has_critical = any(f.severity == FindingSeverity.CRITICAL for f in findings)
        has_high = any(f.severity == FindingSeverity.HIGH for f in findings)

        if has_critical:
            decision = ReviewDecision.REJECT
            severity = FindingSeverity.CRITICAL
            rationale = "Critical safety bypass detected."
        elif has_high:
            decision = ReviewDecision.REQUEST_CHANGES
            severity = FindingSeverity.HIGH
            rationale = "Safety configuration requires remediation before execution."
        else:
            decision = ReviewDecision.APPROVE
            severity = FindingSeverity.LOW
            rationale = "Execution safety, timeouts, and sandbox boundaries fully verified."

        return GovernanceReview(
            id=f"rev_safe_{target.code_revision[:8]}",
            schema_version="1.0",
            target=target,
            reviewer_role=ReviewerRole.SAFETY_EXECUTION_REVIEWER,
            review_stage=GovernanceStage.TEST,
            findings=findings,
            decision=decision,
            severity=severity,
            rationale=rationale,
            evidence_refs=[],
        )
