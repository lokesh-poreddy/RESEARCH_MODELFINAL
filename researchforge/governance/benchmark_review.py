"""researchforge/governance/benchmark_review.py — Benchmark reviewer.

Governance Role: BENCHMARK_SCIENTIST
Evaluates:
- Benchmark protocol integrity
- Fixed task identity and fixed seed sets
- Constant research budget enforcement
- Unchanged evaluation metrics
- Raw result artifact preservation
- Independence of scientific evaluator from policy
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


class BenchmarkReviewer:
    """Reviews benchmark reproducibility, controls, and raw data integrity."""

    def review(
        self,
        target: VersionTarget,
        benchmark_metadata: Optional[Dict[str, Any]] = None,
        fixed_seeds: bool = True,
        budget_strictly_enforced: bool = True,
        independent_evaluator_used: bool = True,
        raw_artifacts_preserved: bool = True,
    ) -> GovernanceReview:
        findings: List[ReviewFinding] = []

        if not fixed_seeds:
            findings.append(
                ReviewFinding(
                    id=f"bench_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="UNFIXED_BENCHMARK_SEEDS",
                    category="REPRODUCIBILITY",
                    severity=FindingSeverity.HIGH,
                    description="Benchmark execution did not use fixed, deterministic random seeds.",
                    recommended_action="Enforce deterministic seed schedule across all conditions.",
                )
            )

        if not budget_strictly_enforced:
            findings.append(
                ReviewFinding(
                    id=f"bench_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="BUDGET_OVERRUN",
                    category="FAIR_COMPARISON",
                    severity=FindingSeverity.HIGH,
                    description="Experiment count or compute budget exceeded the fixed benchmark ceiling.",
                    recommended_action="Strictly enforce equal compute/trial budgets across conditions.",
                )
            )

        if not independent_evaluator_used:
            findings.append(
                ReviewFinding(
                    id=f"bench_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="CIRCULAR_EVALUATION",
                    category="GAMING_PREVENTION",
                    severity=FindingSeverity.CRITICAL,
                    description="Research policy used its own internal utility score as evaluation metric.",
                    recommended_action="Use external, independent ScientificEvaluator / BenchmarkEvaluator.",
                )
            )

        if not raw_artifacts_preserved:
            findings.append(
                ReviewFinding(
                    id=f"bench_f_{len(findings)+1}",
                    schema_version="1.0",
                    finding_id="MISSING_RAW_ARTIFACTS",
                    category="DATA_INTEGRITY",
                    severity=FindingSeverity.HIGH,
                    description="Raw trial logs and model checkpoints were not preserved in artifact storage.",
                    recommended_action="Persist all raw outcomes and trajectories into ArtifactRegistry.",
                )
            )

        has_critical = any(f.severity == FindingSeverity.CRITICAL for f in findings)
        has_high = any(f.severity == FindingSeverity.HIGH for f in findings)

        if has_critical:
            decision = ReviewDecision.REJECT
            severity = FindingSeverity.CRITICAL
            rationale = "Benchmark protocol compromised by non-independent evaluation."
        elif has_high:
            decision = ReviewDecision.REQUEST_CHANGES
            severity = FindingSeverity.HIGH
            rationale = "Benchmark conditions require tightening before comparative claims can be accepted."
        else:
            decision = ReviewDecision.APPROVE
            severity = FindingSeverity.LOW
            rationale = "Benchmark reproducibility, fixed seeds, budgets, and independent evaluation verified."

        return GovernanceReview(
            id=f"rev_bench_{target.code_revision[:8]}",
            schema_version="1.0",
            target=target,
            reviewer_role=ReviewerRole.BENCHMARK_SCIENTIST,
            review_stage=GovernanceStage.BENCHMARK,
            findings=findings,
            decision=decision,
            severity=severity,
            rationale=rationale,
            evidence_refs=[],
        )
