"""researchforge/governance/retrospective.py — Retrospective recorder.

Governance Role: RESEARCH_RETROSPECTIVE
Records structured phase retrospectives as durable, queryable research-system history.
Allows ResearchForge to learn not only from target research experiments, but also
from its own development and engineering trajectory.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .contracts import RetrospectiveRecord, VersionTarget


class RetrospectiveRecorder:
    """Captures and records structured retrospectives at the end of each major milestone."""

    def record(
        self,
        phase: str,
        target: VersionTarget,
        what_changed: List[str],
        what_passed: List[str],
        what_failed: List[str],
        unexpected_behavior: List[str],
        negative_results: List[str],
        architectural_debt: List[str],
        research_insight: str,
        benchmark_insight: str,
        next_upgrade_hypothesis: str,
        provenance_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrospectiveRecord:
        return RetrospectiveRecord(
            id=f"retro_{phase.lower().replace(' ', '_')}_{int(time.time())}",
            schema_version="1.0",
            phase=phase,
            target=target,
            what_changed=list(what_changed),
            what_passed=list(what_passed),
            what_failed=list(what_failed),
            unexpected_behavior=list(unexpected_behavior),
            negative_results=list(negative_results),
            architectural_debt=list(architectural_debt),
            research_insight=research_insight,
            benchmark_insight=benchmark_insight,
            next_upgrade_hypothesis=next_upgrade_hypothesis,
            created_at=time.time(),
            provenance_id=provenance_id,
            metadata=metadata or {},
        )
