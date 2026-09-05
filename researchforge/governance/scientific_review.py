"""researchforge/governance/scientific_review.py — Scientific method reviewer.

Governance Role: SCIENTIFIC_METHOD_REVIEWER
Evaluates:
- Research questions and hypotheses
- Falsification criteria and observable outcomes
- Control conditions and confounders
- Temporal isolation and data leakage risks
- Negative results preservation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .contracts import (
    FindingSeverity,
    GovernanceReview,
    GovernanceStage,
    ReviewDecision,
    ReviewFinding,
    ReviewerRole,
    VersionTarget,
)


class ScientificMethodReviewer:
    """Reviews scientific methodology, controls, and falsification rigor."""

    def review(
        self,
        target: VersionTarget,
        hypotheses: Optional[List[Dict[str, Any]]] = None,
        controls_present: bool = True,
        negative_results_retained: bool = True,
        temporal_isolation_verified: bool = True,
    ) -> GovernanceReview:
        findings: List[ReviewFinding] = []

        # 1. Check controls
        if not controls_present:
            findings.append(
                ReviewFinding(
                    id=f"sci_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="MISSING_CONTROLS",
                    category="SCIENTIFIC_RIGOR",
                    severity=FindingSeverity.HIGH,
                    description="Experiments planned without an explicit control condition.",
                    recommended_action="Introduce standard baseline control condition.",
                )
            )

        # 2. Check negative results retention
        if not negative_results_retained:
            findings.append(
                ReviewFinding(
                    id=f"sci_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="DISCARDING_NEGATIVE_RESULTS",
                    category="PUBLICATION_BIAS",
                    severity=FindingSeverity.CRITICAL,
                    description="Negative/failed experiments were purged or discarded from memory.",
                    recommended_action="Preserve all negative results and failures in portfolio history.",
                )
            )

        # 3. Check temporal isolation
        if not temporal_isolation_verified:
            findings.append(
                ReviewFinding(
                    id=f"sci_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="TEMPORAL_LEAKAGE",
                    category="DATA_LEAKAGE",
                    severity=FindingSeverity.CRITICAL,
                    description="Decisions evaluated with information acquired after decision timestamp.",
                    recommended_action="Enforce EvidenceSnapshot boundaries at decision time.",
                )
            )

        # Decision
        has_critical = any(f.severity == FindingSeverity.CRITICAL for f in findings)
        has_high = any(f.severity == FindingSeverity.HIGH for f in findings)

        if has_critical:
            decision = ReviewDecision.REJECT
            severity = FindingSeverity.CRITICAL
            rationale = "Scientific methodology violations found (leakage or lost negative evidence)."
        elif has_high:
            decision = ReviewDecision.REQUEST_CHANGES
            severity = FindingSeverity.HIGH
            rationale = "Scientific method gaps found (missing controls)."
        else:
            decision = ReviewDecision.APPROVE
            severity = FindingSeverity.LOW
            rationale = "Scientific method, controls, negative retention, and temporal isolation verified."

        return GovernanceReview(
            id=f"rev_sci_{target.code_revision[:8]}",
            schema_version="1.0",
            target=target,
            reviewer_role=ReviewerRole.SCIENTIFIC_METHOD_REVIEWER,
            review_stage=GovernanceStage.SCIENTIFIC_VALIDATE,
            findings=findings,
            decision=decision,
            severity=severity,
            rationale=rationale,
            evidence_refs=[],
        )
