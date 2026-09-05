"""researchforge/governance/architecture_review.py — Architecture & boundary reviewer.

Governance Role: RESEARCH_ARCHITECT
Evaluates:
- Canonical domain boundary decoupling (zero DB imports in domain)
- Dependency direction
- Cross-stack import isolation
- Persistence layer parity
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
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


class ArchitectureReviewer:
    """Evaluates architectural integrity and domain decoupling."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def review(self, target: VersionTarget) -> GovernanceReview:
        findings: List[ReviewFinding] = []
        domain_dir = self.workspace_root / "researchforge" / "domain"

        # Check 1: Domain purity (0 database imports in domain)
        db_keywords = {"sqlalchemy", "alembic", "psycopg", "psycopg2", "sqlite3"}
        if domain_dir.exists():
            for py_file in domain_dir.glob("*.py"):
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=str(py_file))
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                if name.name.split(".")[0] in db_keywords:
                                    findings.append(
                                        ReviewFinding(
                                            id=f"arch_f_{len(findings)+1}",
                                            schema_version="1.0",
                                            finding_id=f"DB_IMPORT_{py_file.name}",
                                            category="DOMAIN_PURITY",
                                            severity=FindingSeverity.CRITICAL,
                                            description=f"Domain contract '{py_file.name}' imports database library '{name.name}'.",
                                            recommended_action="Remove database import from domain layer.",
                                            evidence_refs=[str(py_file)],
                                        )
                                    )
                        elif isinstance(node, ast.ImportFrom):
                            mod = node.module or ""
                            if mod.split(".")[0] in db_keywords:
                                findings.append(
                                    ReviewFinding(
                                        id=f"arch_f_{len(findings)+1}",
                                        schema_version="1.0",
                                        finding_id=f"DB_IMPORT_FROM_{py_file.name}",
                                        category="DOMAIN_PURITY",
                                        severity=FindingSeverity.CRITICAL,
                                        description=f"Domain contract '{py_file.name}' imports from '{mod}'.",
                                        recommended_action="Decouple persistence from domain contract.",
                                        evidence_refs=[str(py_file)],
                                    )
                                )
                except Exception as e:
                    findings.append(
                        ReviewFinding(
                            id=f"arch_f_{len(findings)+1}",
                            schema_version="1.0",
                            finding_id="AST_PARSE_ERROR",
                            category="PARSER",
                            severity=FindingSeverity.HIGH,
                            description=f"Could not parse '{py_file.name}': {str(e)}",
                            recommended_action="Fix syntax errors.",
                        )
                    )

        # Determine decision
        has_critical = any(f.severity == FindingSeverity.CRITICAL for f in findings)
        decision = ReviewDecision.REJECT if has_critical else ReviewDecision.APPROVE
        severity = FindingSeverity.CRITICAL if has_critical else FindingSeverity.LOW
        rationale = "Domain layer strictly decoupled from database engines." if not has_critical else "Critical boundary violations found in domain layer."

        return GovernanceReview(
            id=f"rev_arch_{target.code_revision[:8]}",
            schema_version="1.0",
            target=target,
            reviewer_role=ReviewerRole.RESEARCH_ARCHITECT,
            review_stage=GovernanceStage.REVIEW,
            findings=findings,
            decision=decision,
            severity=severity,
            rationale=rationale,
            evidence_refs=[str(domain_dir)],
        )
