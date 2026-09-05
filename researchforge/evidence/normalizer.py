"""Evidence Normalizer: converts research findings, experimental outcomes,
negative results, replications, trajectories, and literature items into
canonical, typed Evidence.

Scientific Principles (Phase 8):
1. Execution failures and invalid experiments must NOT produce positive evidence.
2. Inconclusive outcomes must remain inconclusive (UNCERTAIN_FROM).
3. Negative results are explicitly preserved as CONTRADICTED_BY evidence.
4. Trajectory evidence references underlying VRDEG node and trajectory IDs rather
   than copying them into opaque text.
5. Evidence quality is multi-dimensional.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from ..domain.claim import EvidenceRelation
from ..domain.evidence import Evidence, EvidenceQuality, EvidenceType
from ..domain.experiment import ExperimentRun, ExperimentSpec, ExecutionStatus
from ..domain.outcome import Outcome
from ..domain.validity import ValidityVerdict


class EvidenceNormalizer:
    """Canonical normalizer transforming raw and structured artifacts into Evidence."""

    @staticmethod
    def normalize_experiment(
        run: ExperimentRun,
        outcome: Optional[Outcome] = None,
        spec: Optional[ExperimentSpec] = None,
        claim_id: Optional[str] = None,
        provenance_id: Optional[str] = None,
        target_metric: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Evidence:
        """Normalize an experimental run and outcome into canonical Evidence.

        Safety & Validity Rules:
        - If run failed or outcome is None: generates negative/contradicting evidence with 0 empirical support.
        - If validity verdict is FAIL: generates contradicting evidence with 0 empirical support.
        - If validity verdict is INCONCLUSIVE/PROVISIONAL: generates UNCERTAIN_FROM evidence.
        - If valid and successful: generates SUPPORTED_BY evidence with empirical support.
        """
        metric_val = 0.0
        if outcome is not None:
            raw_metrics = getattr(outcome, "measured_metrics", None) or getattr(outcome, "metrics", None) or {}
            if target_metric and target_metric in raw_metrics:
                metric_val = float(raw_metrics[target_metric])
            elif raw_metrics:
                metric_val = float(list(raw_metrics.values())[0])

        is_exec_failure = (
            (outcome is None)
            or (getattr(run, "status", None) in (ExecutionStatus.FAILED, "FAILED"))
            or (getattr(run, "failure_info", None) is not None)
            or (getattr(outcome, "success", None) is False)
        )
        verdict = (
            outcome.validity.verdict
            if (outcome and outcome.validity)
            else (ValidityVerdict.FAIL if is_exec_failure else ValidityVerdict.INCONCLUSIVE)
        )

        if is_exec_failure:
            ev_type = EvidenceType.NEGATIVE.value
            relation = EvidenceRelation.CONTRADICTED_BY.value
            quality = EvidenceQuality(
                source_reliability=1.0,
                relevance=1.0,
                empirical_support=0.0,
                contradiction_status="none",
                provenance_completeness=1.0,
            )
            err_msg = ""
            if getattr(run, "failure_info", None):
                err_msg = str(run.failure_info.get("message") or run.failure_info.get("error") or run.failure_info)
            snippet = f"Execution failure: {err_msg or getattr(run, 'status', 'FAILED')}"
        elif verdict == ValidityVerdict.FAIL:
            ev_type = EvidenceType.EXPERIMENTAL.value
            relation = EvidenceRelation.CONTRADICTED_BY.value
            quality = EvidenceQuality(
                source_reliability=1.0,
                relevance=1.0,
                empirical_support=0.0,
                contradiction_status="none",
                provenance_completeness=1.0,
            )
            details = getattr(outcome.validity, "details", None) if outcome and outcome.validity else None
            snippet = f"Invalid experiment: {details or 'verdict FAIL'}"
        elif verdict in (ValidityVerdict.INCONCLUSIVE, ValidityVerdict.PROVISIONAL, ValidityVerdict.REQUIRES_HUMAN_REVIEW):
            ev_type = EvidenceType.EXPERIMENTAL.value
            relation = EvidenceRelation.UNCERTAIN_FROM.value
            quality = EvidenceQuality(
                source_reliability=1.0,
                relevance=1.0,
                empirical_support=0.5,
                contradiction_status="none",
                provenance_completeness=1.0,
            )
            details = getattr(outcome.validity, "details", None) if outcome and outcome.validity else None
            snippet = f"Inconclusive experiment: {details or 'verdict INCONCLUSIVE'}"
        else:  # PASS
            ev_type = EvidenceType.EXPERIMENTAL.value
            relation = EvidenceRelation.SUPPORTED_BY.value
            quality = EvidenceQuality(
                source_reliability=1.0,
                relevance=1.0,
                empirical_support=round(max(0.0, min(1.0, metric_val)), 4),
                contradiction_status="none",
                provenance_completeness=1.0,
            )
            snippet = f"Valid experiment: metric {metric_val}"

        spec_id = spec.id if spec else getattr(run, "spec_id", getattr(run, "experiment_spec_id", ""))
        outcome_id = outcome.id if outcome else "no_outcome"
        ev_id = f"ev_exp_{run.id}_{outcome_id}"

        originating_nodes = [run.id]
        if outcome:
            originating_nodes.append(outcome.id)
        if spec_id:
            originating_nodes.append(spec_id)

        return Evidence(
            id=ev_id,
            schema_version="1",
            source="experiment",
            source_id=run.id,
            source_type="experiment",
            evidence_type=ev_type,
            claim_id=claim_id,
            relation=relation,
            provenance_id=provenance_id or (getattr(outcome, "provenance_id", None) if outcome else None),
            snippet=snippet,
            quality=quality,
            originating_run_ids=[run.id],
            originating_outcome_ids=[outcome.id] if outcome else [],
            originating_experiment_ids=[spec_id] if spec_id else [],
            originating_vrdeg_node_ids=originating_nodes,
            created_at=time.time(),
            metadata={
                "metric_value": metric_val,
                "validity_verdict": verdict.value if isinstance(verdict, ValidityVerdict) else str(verdict),
                "failure_category": getattr(outcome, "failure_category", None),
            },
        )

    @staticmethod
    def normalize_negative_result(
        source_id: Optional[str] = None,
        run_or_spec_id: Optional[str] = None,
        failure_category: str = "negative_result",
        finding_summary: Optional[str] = None,
        metric: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
        claim_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        provenance_id: Optional[str] = None,
        originating_run_ids: Optional[List[str]] = None,
    ) -> Evidence:
        """Explicitly capture and normalize a negative result or failure into retrievable evidence."""
        src_id = source_id or run_or_spec_id or "unknown_run"
        ev_id = f"ev_neg_{src_id}_{failure_category}"
        quality = EvidenceQuality(
            source_reliability=1.0,
            relevance=1.0,
            empirical_support=0.0,
            contradiction_status="none",
            provenance_completeness=1.0,
        )
        runs = originating_run_ids or ([src_id] if src_id else [])
        snippet = finding_summary or f"Negative result in {src_id}: category {failure_category}"
        return Evidence(
            id=ev_id,
            schema_version="1",
            source="negative_result",
            source_id=src_id,
            source_type="failure",
            evidence_type=EvidenceType.NEGATIVE.value,
            claim_id=claim_id,
            relation=EvidenceRelation.CONTRADICTED_BY.value,
            provenance_id=provenance_id,
            snippet=snippet,
            quality=quality,
            originating_run_ids=runs,
            originating_vrdeg_node_ids=runs,
            created_at=time.time(),
            metadata={
                "failure_category": failure_category,
                "metric": metric,
                "metrics": metrics or {},
                "details": details or {},
            },
        )

    @staticmethod
    def normalize_trajectory(
        trajectory_record: Optional[Any] = None,
        trajectory_id: Optional[str] = None,
        node_ids: Optional[List[str]] = None,
        trajectory_summary: Optional[str] = None,
        claim_id: Optional[str] = None,
        relation: str | EvidenceRelation = EvidenceRelation.SUPPORTED_BY,
        success: bool = True,
        metric: float = 0.0,
        provenance_id: Optional[str] = None,
    ) -> Evidence:
        """Convert a trajectory record into Evidence without destroying underlying VRDEG references."""
        traj_id = trajectory_id or getattr(trajectory_record, "id", "traj_001")
        if trajectory_record is not None:
            succ = getattr(trajectory_record, "success", False)
            met = float(getattr(trajectory_record, "metric", 0.0))
        else:
            succ = success
            met = metric

        rel_str = relation.value if isinstance(relation, EvidenceRelation) else str(relation)
        quality = EvidenceQuality(
            source_reliability=1.0,
            relevance=1.0,
            empirical_support=round(max(0.0, min(1.0, met)), 4),
            contradiction_status="none",
            provenance_completeness=1.0,
        )

        originating_nodes = list(node_ids or [])
        if trajectory_record is not None:
            for attr in ("finding_id", "outcome_id", "run_id", "hypothesis_id"):
                val = getattr(trajectory_record, attr, None)
                if val and val not in originating_nodes:
                    originating_nodes.append(val)

        summary = trajectory_summary or f"Trajectory {traj_id} (success={succ}, metric={met})"

        return Evidence(
            id=f"ev_traj_{traj_id}",
            schema_version="1",
            source="trajectory_memory",
            source_id=traj_id,
            source_type="trajectory",
            evidence_type=EvidenceType.TRAJECTORY.value,
            claim_id=claim_id,
            relation=rel_str,
            provenance_id=provenance_id or getattr(trajectory_record, "provenance_id", None),
            snippet=summary,
            quality=quality,
            originating_trajectory_ids=[traj_id],
            originating_vrdeg_node_ids=originating_nodes,
            created_at=time.time(),
            metadata={
                "strategy": getattr(trajectory_record, "strategy", ""),
                "model_type": getattr(trajectory_record, "child_model_type", ""),
                "success": succ,
                "metric": met,
            },
        )

    @staticmethod
    def normalize_literature(
        source: Optional[str] = None,
        source_id: Optional[str] = None,
        title: Optional[str] = None,
        content: Optional[str] = None,
        summary: Optional[str] = None,
        relevance_score: Optional[float] = None,
        relevance: Optional[float] = None,
        citation_key: Optional[str] = None,
        doi_or_url: Optional[str] = None,
        claim_id: Optional[str] = None,
        relation: str | EvidenceRelation = EvidenceRelation.SUPPORTED_BY,
        source_reliability: float = 0.9,
        provenance_id: Optional[str] = None,
    ) -> Evidence:
        """Normalize an academic or literature finding into canonical Evidence."""
        rel_score = relevance if relevance is not None else (relevance_score if relevance_score is not None else 1.0)
        t = title or "Literature Finding"
        s = source or source_id or "literature"
        body = summary or content or ""
        rel_str = relation.value if isinstance(relation, EvidenceRelation) else str(relation)
        key_str = f"{s}:{t}:{doi_or_url or ''}:{source_id or ''}"
        ev_id = f"ev_lit_{hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:12]}"

        quality = EvidenceQuality(
            source_reliability=source_reliability,
            relevance=rel_score,
            empirical_support=0.8,
            freshness=1.0,
            provenance_completeness=1.0 if (citation_key or doi_or_url) else 0.7,
        )

        return Evidence(
            id=ev_id,
            schema_version="1",
            source=s,
            source_id=source_id or citation_key or doi_or_url or t,
            source_type="literature",
            evidence_type=EvidenceType.LITERATURE.value,
            claim_id=claim_id,
            relation=rel_str,
            provenance_id=provenance_id,
            source_url=doi_or_url,
            snippet=body[:500] if body else None,
            quality=quality,
            created_at=time.time(),
            metadata={
                "title": t,
                "citation_key": citation_key,
            },
        )

    @staticmethod
    def normalize_replication(
        original_evidence_id: str,
        replication_run: Optional[ExperimentRun] = None,
        replication_run_id: Optional[str] = None,
        replication_outcome: Optional[Outcome] = None,
        replicates_success: Optional[bool] = None,
        replicated_successfully: Optional[bool] = None,
        metrics_comparison: Optional[Dict[str, Any]] = None,
        claim_id: Optional[str] = None,
        provenance_id: Optional[str] = None,
    ) -> Evidence:
        """Normalize an independent replication attempt into canonical Evidence."""
        run_id = replication_run.id if replication_run else (replication_run_id or "rep_run")
        outcome_id = replication_outcome.id if replication_outcome else "rep_outcome"
        is_succ = (
            replicated_successfully if replicated_successfully is not None
            else (replicates_success if replicates_success is not None else True)
        )

        ev_id = f"ev_rep_{original_evidence_id}_{run_id}"
        rep_status = "replicated" if is_succ else "failed_replication"
        rel_str = (
            EvidenceRelation.SUPPORTED_BY.value if is_succ
            else EvidenceRelation.CONTRADICTED_BY.value
        )
        quality = EvidenceQuality(
            source_reliability=1.0,
            relevance=1.0,
            empirical_support=1.0 if is_succ else 0.0,
            replication_status=rep_status,
            contradiction_status="has_contradiction" if not is_succ else "none",
            provenance_completeness=1.0,
        )

        originating_nodes = [run_id]
        if replication_outcome:
            originating_nodes.append(replication_outcome.id)

        return Evidence(
            id=ev_id,
            schema_version="1",
            source="replication_experiment",
            source_id=run_id,
            source_type="replication",
            evidence_type=EvidenceType.REPLICATION.value,
            claim_id=claim_id,
            relation=rel_str,
            provenance_id=provenance_id or (getattr(replication_outcome, "provenance_id", None) if replication_outcome else None),
            snippet=f"Replication of {original_evidence_id} (success={is_succ})",
            quality=quality,
            originating_run_ids=[run_id],
            originating_outcome_ids=[outcome_id] if replication_outcome else [],
            originating_vrdeg_node_ids=originating_nodes,
            created_at=time.time(),
            metadata={
                "original_evidence_id": original_evidence_id,
                "replicates_success": is_succ,
                "metrics_comparison": metrics_comparison or {},
            },
        )
