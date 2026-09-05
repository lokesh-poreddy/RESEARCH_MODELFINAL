"""researchforge/policy/telemetry.py — Research Policy Observation & Outcome Telemetry.

Phase 9A Invariants:
1. Observational Evaluator Output: Research progress is an observational metric output by
   independent evaluators, NOT a causal attribution claim.
2. Controlled Progress Statuses: Explicitly classifies progress into:
   - VALID_EMPIRICAL_GAIN
   - NO_CHANGE
   - DEGRADATION
   - INCONCLUSIVE
   - INVALID
   - NOT_ASSESSED
3. Timezone-aware UTC timestamp representation.
4. Complete auditability connecting:
   predicted utility -> actual outcome -> baseline outcome -> delta -> validity -> research progress.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ..domain.base import DomainObject, _canonical_json
from ..domain.validity import ValidityVerdict
from .decision_record import PolicyDecisionRecord


class ResearchProgressStatus(str, Enum):
    """Categorical classification of observed research progress."""
    VALID_EMPIRICAL_GAIN = "VALID_EMPIRICAL_GAIN"
    NO_CHANGE = "NO_CHANGE"
    DEGRADATION = "DEGRADATION"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"
    NOT_ASSESSED = "NOT_ASSESSED"


@dataclass(frozen=True)
class ResearchOutcomeObservation(DomainObject):
    """Connects a policy decision to its downstream empirical outcome, validity, and progress."""
    id: str
    schema_version: str
    policy_decision_id: str
    decision_id: str
    action_id: str
    spec_id: str
    run_id: str
    outcome_id: str

    predicted_utility: float
    actual_metric: Optional[float]
    baseline_metric: Optional[float]
    delta: Optional[float]

    validity_verdict: ValidityVerdict
    research_progress_status: ResearchProgressStatus
    research_progress_value: Optional[float]  # Only populated when a valid evaluator produces one
    research_progress_basis: str

    observation_timestamp: str  # ISO-8601 UTC string
    provenance_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def observation_fingerprint(self) -> str:
        """Deterministic fingerprint of this outcome observation."""
        payload = {
            "schema_version": self.schema_version,
            "policy_decision_id": self.policy_decision_id,
            "decision_id": self.decision_id,
            "action_id": self.action_id,
            "run_id": self.run_id,
            "outcome_id": self.outcome_id,
            "predicted_utility": round(self.predicted_utility, 6),
            "actual_metric": round(self.actual_metric, 6) if self.actual_metric is not None else None,
            "baseline_metric": round(self.baseline_metric, 6) if self.baseline_metric is not None else None,
            "delta": round(self.delta, 6) if self.delta is not None else None,
            "validity_verdict": self.validity_verdict.value if isinstance(self.validity_verdict, ValidityVerdict) else str(self.validity_verdict),
            "research_progress_status": self.research_progress_status.value if isinstance(self.research_progress_status, ResearchProgressStatus) else str(self.research_progress_status),
            "research_progress_value": round(self.research_progress_value, 6) if self.research_progress_value is not None else None,
        }
        j = _canonical_json(payload)
        return hashlib.sha256(j.encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        policy_decision: PolicyDecisionRecord,
        spec_id: str,
        run_id: str,
        outcome_id: str,
        actual_metric: Optional[float],
        baseline_metric: Optional[float] = None,
        validity_verdict: ValidityVerdict = ValidityVerdict.PASS,
        evaluator_basis: str = "StandardScientificEvaluator",
        observation_timestamp: Optional[datetime] = None,
        provenance_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ResearchOutcomeObservation":
        """Deterministic constructor binding policy decision to outcome and validity."""
        ts = observation_timestamp or datetime.now(timezone.utc)
        ts_iso = ts.isoformat()

        # Compute delta if metrics available
        delta = None
        if actual_metric is not None and baseline_metric is not None:
            delta = round(actual_metric - baseline_metric, 6)

        # Classify progress status with strict validity gating
        # Observational evaluator output only; no causal claims
        if validity_verdict != ValidityVerdict.PASS:
            progress_status = ResearchProgressStatus.INVALID
            progress_val = None
            basis_desc = f"{evaluator_basis}: validity verdict is {validity_verdict.value}; progress not certified."
        elif actual_metric is None or baseline_metric is None or delta is None:
            progress_status = ResearchProgressStatus.NOT_ASSESSED
            progress_val = None
            basis_desc = f"{evaluator_basis}: baseline or actual metric unavailable; comparative progress not assessed."
        elif delta > 1e-4:
            progress_status = ResearchProgressStatus.VALID_EMPIRICAL_GAIN
            progress_val = delta
            basis_desc = f"{evaluator_basis}: valid empirical gain of {delta:+.6f} over baseline ({baseline_metric:.4f} -> {actual_metric:.4f})."
        elif abs(delta) <= 1e-4:
            progress_status = ResearchProgressStatus.NO_CHANGE
            progress_val = 0.0
            basis_desc = f"{evaluator_basis}: valid empirical parity with baseline ({baseline_metric:.4f} -> {actual_metric:.4f})."
        else:
            progress_status = ResearchProgressStatus.DEGRADATION
            progress_val = delta
            basis_desc = f"{evaluator_basis}: valid empirical degradation of {delta:+.6f} below baseline ({baseline_metric:.4f} -> {actual_metric:.4f})."

        obs_id = f"obs_{policy_decision.id}_{outcome_id}"

        return cls(
            id=obs_id,
            schema_version="1.0",
            policy_decision_id=policy_decision.id,
            decision_id=policy_decision.decision_id,
            action_id=policy_decision.selected_action_id,
            spec_id=spec_id,
            run_id=run_id,
            outcome_id=outcome_id,
            predicted_utility=policy_decision.predicted_utility,
            actual_metric=actual_metric,
            baseline_metric=baseline_metric,
            delta=delta,
            validity_verdict=validity_verdict,
            research_progress_status=progress_status,
            research_progress_value=progress_val,
            research_progress_basis=basis_desc,
            observation_timestamp=ts_iso,
            provenance_id=provenance_id or policy_decision.provenance_id,
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ResearchOutcomeObservation":
        d = dict(obj)
        if "validity_verdict" in d and isinstance(d["validity_verdict"], str):
            d["validity_verdict"] = ValidityVerdict(d["validity_verdict"])
        if "research_progress_status" in d and isinstance(d["research_progress_status"], str):
            d["research_progress_status"] = ResearchProgressStatus(d["research_progress_status"])
        return cls(**d)
