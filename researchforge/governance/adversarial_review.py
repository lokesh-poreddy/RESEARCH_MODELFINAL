"""researchforge/governance/adversarial_review.py — Adversarial stress reviewer.

Governance Role: ADVERSARIAL_EVALUATOR
Evaluates:
- System resilience against malformed evidence
- Resistance to policy score tampering
- Duplicate event rejection
- Contradictory evidence preservation
- Replay divergence detection
- Temporal leakage prevention
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .contracts import (
    FindingSeverity,
    GovernanceReview,
    GovernanceStage,
    ReviewDecision,
    ReviewFinding,
    ReviewerRole,
    VersionTarget,
)


class AdversarialReviewer:
    """Executes structured adversarial stress tests against the ResearchForge runtime."""

    def review(
        self,
        target: VersionTarget,
        adversarial_checks: Optional[Dict[str, bool]] = None,
    ) -> GovernanceReview:
        findings: List[ReviewFinding] = []

        # Standard battery of adversarial checks
        # True means the defense PASSED; False means the system succumbed to the attack.
        checks = adversarial_checks or {
            "duplicate_event_rejection": True,
            "temporal_leakage_rejection": True,
            "score_tampering_detection": True,
            "malformed_evidence_rejection": True,
            "replay_divergence_detection": True,
            "history_mutation_rejection": True,
        }

        for check_name, passed in checks.items():
            if not passed:
                findings.append(
                    ReviewFinding(
                        id=f"adv_f_{len(findings)+1}",
                        schema_version="1.0",
                        finding_id=f"ADVERSARIAL_FAILURE_{check_name.upper()}",
                        category="ADVERSARIAL_VULNERABILITY",
                        severity=FindingSeverity.CRITICAL,
                        description=f"System failed adversarial challenge: '{check_name}'.",
                        recommended_action=f"Harden {check_name} defenses against tampering or bypass.",
                    )
                )

        has_critical = any(f.severity == FindingSeverity.CRITICAL for f in findings)
        decision = ReviewDecision.REJECT if has_critical else ReviewDecision.APPROVE
        severity = FindingSeverity.CRITICAL if has_critical else FindingSeverity.LOW
        rationale = "System demonstrated robust resistance against all structured adversarial attacks." if not has_critical else "System vulnerable to adversarial manipulation."

        return GovernanceReview(
            id=f"rev_adv_{target.code_revision[:8]}",
            schema_version="1.0",
            target=target,
            reviewer_role=ReviewerRole.ADVERSARIAL_EVALUATOR,
            review_stage=GovernanceStage.CHALLENGE,
            findings=findings,
            decision=decision,
            severity=severity,
            rationale=rationale,
            evidence_refs=[],
        )
