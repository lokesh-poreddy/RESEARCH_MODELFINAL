"""researchforge/policy/research_policy.py — Transparent, auditable ResearchPolicy.

Scientific & Architectural Requirements (Phase 9):
1. Score decomposition: inspectable weights and component utilities.
2. Temporal isolation: strictly bounded by EvidenceSnapshot.
3. Failure-aware: repeated contextual failures increase risk without universal bans.
4. Negative transfer classification.
5. Emits both canonical domain Decision and detailed PolicyDecisionRecord.
6. Deterministic replay: same inputs produce identical decision fingerprint.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..domain.action import ActionType, ResearchAction
from ..domain.decision import Decision
from ..domain.state import ResearchState
from ..evidence.snapshot import EvidenceSnapshot
from ..memory.adaptive_trajectory import AdaptiveTrajectoryMemory
from .config import PolicyAblationMode, PolicyConfig
from .decision_record import PolicyDecisionRecord
from .score_decomposition import ScoreDecomposition, TransferAssessment


class ResearchPolicy:
    """Transparent research decision policy."""

    def __init__(
        self,
        config: Optional[PolicyConfig] = None,
        memory: Optional[AdaptiveTrajectoryMemory] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = config or PolicyConfig.baseline_v1()
        self.memory = memory
        self.rng = rng or random.Random(42)

    @property
    def policy_version(self) -> str:
        return self.config.policy_version

    @property
    def policy_fingerprint(self) -> str:
        return self.config.fingerprint()

    def evaluate_candidates(
        self,
        state: ResearchState,
        candidate_actions: List[ResearchAction],
        snapshot: EvidenceSnapshot,
        decision_timestamp: Optional[float] = None,
    ) -> Tuple[Decision, PolicyDecisionRecord]:
        """Evaluate candidate actions and select the optimal research action under the configured policy."""
        if not candidate_actions:
            raise ValueError("Cannot evaluate policy without candidate actions.")

        ts = decision_timestamp if decision_timestamp is not None else time.time()
        weights = self.config.weights_dict()
        mode = self.config.ablation_mode

        decompositions: Dict[str, ScoreDecomposition] = {}

        # 1. Evaluate each candidate action under the configured mode
        for action in candidate_actions:
            raw_scores = self._score_action(action, state, snapshot, mode)
            transfer_class = self._classify_transfer(action, state, snapshot)
            decomp = ScoreDecomposition.compute(
                action_id=action.id,
                action_type=action.action_type if isinstance(action.action_type, str) else action.action_type.value,
                raw_scores=raw_scores,
                weights=weights,
                transfer_assessment=transfer_class,
                metadata={"ablation_mode": mode.value},
            )
            decompositions[action.id] = decomp

        # 2. Select best action based on ablation mode
        if mode == PolicyAblationMode.RANDOM:
            selected_action = self.rng.choice(candidate_actions)
        else:
            # Sort deterministically: highest utility first, break ties by action id
            sorted_candidates = sorted(
                candidate_actions,
                key=lambda a: (decompositions[a.id].total_utility, a.id),
                reverse=True,
            )
            selected_action = sorted_candidates[0]

        selected_id = selected_action.id
        selected_decomp = decompositions[selected_id]
        rejected_ids = [a.id for a in candidate_actions if a.id != selected_id]

        decision_id = f"dec_{selected_id}_{int(ts)}"

        # 3. Construct Domain-Level Decision
        domain_decision = Decision(
            id=decision_id,
            schema_version="1.0",
            research_state_fingerprint=state.fingerprint(),
            rsg_id=state.selected_rsg_id,
            hypothesis_id=selected_action.target_context.get("hypothesis_id"),
            selected_tmg_id=selected_action.target_context.get("target_model_id"),
            selected_operator=selected_action.target_context.get("strategy") or selected_action.action_type,
            decision_reason=selected_decomp.explanation,
            evidence_refs=list(snapshot.evidence_ids),
            expected_information_gain=selected_decomp.information_gain,
            expected_performance_gain=selected_decomp.performance,
            estimated_cost=selected_decomp.cost,
            estimated_failure_risk=selected_decomp.failure_risk,
            exploration_score=selected_decomp.novelty,
            novelty_score=selected_decomp.novelty,
            confidence=max(0.1, 1.0 - selected_decomp.failure_risk),
            decision_timestamp=str(ts),
            policy_version=self.policy_version,
        )

        # 4. Construct Detailed PolicyDecisionRecord
        credit_assignment_meta = {
            "prior_state_id": state.id,
            "prior_state_fingerprint": state.fingerprint(),
            "selected_action_id": selected_id,
            "predicted_gain": selected_decomp.performance,
            "expected_utility": selected_decomp.total_utility,
            "evaluated_at": ts,
        }

        policy_record = PolicyDecisionRecord(
            id=f"pdr_{decision_id}",
            schema_version="1.0",
            decision_id=decision_id,
            research_state_fingerprint=state.fingerprint(),
            evidence_snapshot_id=snapshot.id,
            policy_version=self.policy_version,
            policy_fingerprint=self.policy_fingerprint,
            selected_action_id=selected_id,
            selected_action=selected_action,
            rejected_action_ids=rejected_ids,
            score_decompositions=decompositions,
            policy_weights=weights,
            decision_timestamp=ts,
            explanation=selected_decomp.explanation,
            available_memory_ref=getattr(self.memory, "id", None) if self.memory else None,
            provenance_id=selected_action.provenance_id,
            credit_assignment_meta=credit_assignment_meta,
            evidence_snapshot_fingerprint=getattr(snapshot, "fingerprint", lambda: snapshot.id)() if callable(getattr(snapshot, "fingerprint", None)) else snapshot.id,
            candidate_actions_list=candidate_actions,
            baseline_score=0.5,
        )

        return domain_decision, policy_record

    def _score_action(
        self,
        action: ResearchAction,
        state: ResearchState,
        snapshot: EvidenceSnapshot,
        mode: PolicyAblationMode,
    ) -> Dict[str, float]:
        """Compute inspectable raw score components."""
        ctx = action.target_context or {}
        strategy = ctx.get("strategy", "default")
        model_family = ctx.get("model_family", "mlp")

        if mode == PolicyAblationMode.STATIC_BASELINE:
            # Fixed baseline scores ignoring memory and dynamic evidence
            return {
                "performance": 0.5,
                "information_gain": 0.3,
                "novelty": 0.2,
                "evidence": 0.1,
                "transfer": 0.0,
                "failure_risk": 0.1,
                "redundancy": 0.0,
                "cost": 0.2,
            }

        # Estimate failure risk context-sensitively using memory if available
        failure_risk = 0.05
        transfer_score = 0.5
        performance_est = 0.6
        novelty_score = 0.4

        if self.memory is not None:
            raw_bucket = str(ctx.get("capacity_bucket", "medium")).lower()
            bucket = raw_bucket if raw_bucket in ("low", "medium", "high") else "medium"

            # Check if similar trajectory recently failed
            fail_check = self.memory.similar_trajectory_recently_failed(
                strategy=strategy,
                model_type=model_family,
                capacity_bucket=bucket,
                min_samples=1,
            )
            if fail_check.has_failed_majority and fail_check.context_level >= 1:
                # Repeated context-specific failure raises risk proportionally to frequency
                failure_risk = min(0.95, 0.35 + 0.15 * len(fail_check.failed_records))
            elif fail_check.has_failed_majority:
                # Global fallback failure provides a weak background prior without context-specific penalty
                failure_risk = 0.2
            else:
                failure_risk = 0.1

            # Query contextual retrieval result
            retrieval = self.memory.query_context(
                strategy=strategy,
                model_type=model_family,
                capacity_bucket=bucket,
            )
            if retrieval.sample_count > 0:
                performance_est = retrieval.mean_metric if retrieval.mean_metric is not None else 0.6
                transfer_score = retrieval.success_rate
                # Novelty decreases as we accumulate more samples of this exact strategy
                novelty_score = max(0.05, 1.0 / (1.0 + math.log1p(retrieval.sample_count)))

        # Evidence support from decision-time snapshot
        # Strictly check evidence in snapshot
        matching_evidence_count = 0
        if snapshot.evidence_items:
            for ev in snapshot.evidence_items:
                if ev.source_id == action.id or ev.claim_id == ctx.get("hypothesis_id"):
                    matching_evidence_count += 1
        evidence_score = min(1.0, matching_evidence_count * 0.25)

        # UCB exploration bonus
        if mode == PolicyAblationMode.UCB:
            novelty_score = min(1.0, novelty_score * 1.5)

        # Redundancy penalty: higher if action already attempted recently in state
        redundancy = 0.0
        if state.recent_experiment_refs and action.id in state.recent_experiment_refs:
            redundancy = 0.7

        # Cost: scaled by parameters
        cost = float(action.parameters.get("cost", 0.2)) if action.parameters else 0.2

        info_gain = max(0.1, novelty_score * (1.0 - failure_risk))

        return {
            "performance": min(1.0, max(0.0, performance_est)),
            "information_gain": min(1.0, max(0.0, info_gain)),
            "novelty": min(1.0, max(0.0, novelty_score)),
            "evidence": min(1.0, max(0.0, evidence_score)),
            "transfer": min(1.0, max(0.0, transfer_score)),
            "failure_risk": min(1.0, max(0.0, failure_risk)),
            "redundancy": min(1.0, max(0.0, redundancy)),
            "cost": min(1.0, max(0.0, cost)),
        }

    def _classify_transfer(
        self,
        action: ResearchAction,
        state: ResearchState,
        snapshot: EvidenceSnapshot,
    ) -> TransferAssessment:
        """Classify cross-context transfer potential."""
        ctx = action.target_context or {}
        if not self.memory:
            return TransferAssessment.INSUFFICIENT_EVIDENCE

        raw_bucket = str(ctx.get("capacity_bucket", "medium")).lower()
        bucket = raw_bucket if raw_bucket in ("low", "medium", "high") else "medium"
        retrieval = self.memory.query_context(
            strategy=ctx.get("strategy", "default"),
            model_type=ctx.get("model_family", "mlp"),
            capacity_bucket=bucket,
        )
        if retrieval.sample_count < 2:
            return TransferAssessment.INSUFFICIENT_EVIDENCE

        if retrieval.success_rate >= 0.7:
            return TransferAssessment.POSITIVE_TRANSFER
        elif retrieval.success_rate <= 0.3:
            return TransferAssessment.NEGATIVE_TRANSFER
        else:
            return TransferAssessment.UNCERTAIN_TRANSFER
