"""researchforge/transfer/guard.py — TransferGuard selective experience gating.

Phase 13 (RF-1.0.0-beta.1):
Enforces deterministic gating policies (STRICT, SELECTIVE, DISABLED).
Protects against false-positive suppression: useful cross-family transfer
with positive historical empirical evidence is preserved.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .affinity import (
    TaskProfile,
    compute_historical_transfer_effect,
    compute_prior_affinity,
    compute_transfer_score,
)
from .types import (
    ALLOWED_SELECTIVE_CATEGORIES,
    RESTRICTED_SELECTIVE_CATEGORIES,
    TransferDecisionRecord,
    TransferGuardMode,
    TransferableKnowledgeCategory,
)

logger = logging.getLogger(__name__)


class TransferGuard:
    """Selective experience gating and negative transfer suppression."""

    def __init__(
        self,
        mode: TransferGuardMode = TransferGuardMode.SELECTIVE,
        suppression_threshold: float = -0.25,
        false_positive_protection: bool = True,
    ) -> None:
        self.mode = mode
        self.suppression_threshold = suppression_threshold
        self.false_positive_protection = false_positive_protection
        self._decision_history: List[TransferDecisionRecord] = []

    @property
    def decision_history(self) -> List[TransferDecisionRecord]:
        return list(self._decision_history)

    def filter_memories(
        self,
        memories: List[Dict[str, Any]],
        source_profile: TaskProfile,
        target_profile: TaskProfile,
        transfer_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], TransferDecisionRecord]:
        """Apply gating policy to candidate memories transferred from source to target task.
        
        Each memory dictionary is expected to have an 'id' and optionally 'knowledge_category'
        (defaults to TASK_SPECIFIC_HYPOTHESIS if untyped).
        """
        if not memories:
            decision = TransferDecisionRecord(
                source_task_id=source_profile.task_id,
                target_task_id=target_profile.task_id,
                source_family=source_profile.task_family,
                target_family=target_profile.task_family,
                prior_affinity=1.0 if source_profile.task_id == target_profile.task_id else 0.5,
                historical_transfer_effect=None,
                evidence_volume=0,
                transfer_score=0.0,
                guard_mode=self.mode,
                action_taken="ALLOW_ALL",
                allowed_memory_ids=[],
                filtered_memory_ids=[],
                rationale="Empty candidate memory set",
            )
            self._decision_history.append(decision)
            return [], decision

        # 1. Compute prior affinity and empirical effect
        prior_aff = compute_prior_affinity(source_profile, target_profile)
        hist_effect, ev_vol = compute_historical_transfer_effect(
            transfer_history or [],
            source_profile.task_family,
            target_profile.task_family,
        )
        score = compute_transfer_score(prior_aff, hist_effect, ev_vol)

        all_ids = [m.get("id", f"mem_{i}") for i, m in enumerate(memories)]

        # 2. DISABLED Mode
        if self.mode == TransferGuardMode.DISABLED:
            decision = TransferDecisionRecord(
                source_task_id=source_profile.task_id,
                target_task_id=target_profile.task_id,
                source_family=source_profile.task_family,
                target_family=target_profile.task_family,
                prior_affinity=prior_aff,
                historical_transfer_effect=hist_effect,
                evidence_volume=ev_vol,
                transfer_score=score,
                guard_mode=self.mode,
                action_taken="ALLOW_ALL",
                allowed_memory_ids=all_ids,
                filtered_memory_ids=[],
                rationale="TransferGuard is DISABLED; all memories passed unconditionally.",
            )
            self._decision_history.append(decision)
            return memories, decision

        # 3. Check for False-Positive Protection
        # If historical transfer effect is positive with evidence, permit cross-family transfer
        is_cross_family = source_profile.task_family != target_profile.task_family
        if (
            self.false_positive_protection
            and is_cross_family
            and hist_effect is not None
            and hist_effect > 0.05
            and ev_vol >= 2
        ):
            decision = TransferDecisionRecord(
                source_task_id=source_profile.task_id,
                target_task_id=target_profile.task_id,
                source_family=source_profile.task_family,
                target_family=target_profile.task_family,
                prior_affinity=prior_aff,
                historical_transfer_effect=hist_effect,
                evidence_volume=ev_vol,
                transfer_score=score,
                guard_mode=self.mode,
                action_taken="ALLOW_EMPIRICAL_EXCEPTION",
                allowed_memory_ids=all_ids,
                filtered_memory_ids=[],
                rationale=(
                    f"False-positive protection: positive historical transfer effect "
                    f"({hist_effect:+.3f}, N={ev_vol}) overrides cross-family barrier."
                ),
            )
            self._decision_history.append(decision)
            return memories, decision

        # 4. STRICT Mode (complete isolation)
        # Suppress everything if in STRICT mode or if empirical score indicates severe negative transfer
        if self.mode == TransferGuardMode.STRICT or score < self.suppression_threshold:
            action = "SUPPRESS_ALL"
            rat = (
                f"STRICT mode isolation active (or severe negative score {score:+.3f} < {self.suppression_threshold}). "
                f"Suppressed all {len(memories)} prior memories."
            )
            decision = TransferDecisionRecord(
                source_task_id=source_profile.task_id,
                target_task_id=target_profile.task_id,
                source_family=source_profile.task_family,
                target_family=target_profile.task_family,
                prior_affinity=prior_aff,
                historical_transfer_effect=hist_effect,
                evidence_volume=ev_vol,
                transfer_score=score,
                guard_mode=self.mode,
                action_taken=action,
                allowed_memory_ids=[],
                filtered_memory_ids=all_ids,
                rationale=rat,
            )
            self._decision_history.append(decision)
            return [], decision

        # 5. SELECTIVE Mode (typed knowledge categorization)
        allowed_memories: List[Dict[str, Any]] = []
        allowed_ids: List[str] = []
        filtered_ids: List[str] = []

        for m in memories:
            raw_cat = m.get("knowledge_category")
            if isinstance(raw_cat, str):
                try:
                    cat = TransferableKnowledgeCategory(raw_cat)
                except ValueError:
                    cat = TransferableKnowledgeCategory.TASK_SPECIFIC_HYPOTHESIS
            elif isinstance(raw_cat, TransferableKnowledgeCategory):
                cat = raw_cat
            else:
                # Default untyped memory to task-specific hypothesis
                cat = TransferableKnowledgeCategory.TASK_SPECIFIC_HYPOTHESIS

            m_id = m.get("id", "unknown")

            # Same task or same family: allow all unless explicit negative effect
            if not is_cross_family:
                allowed_memories.append(m)
                allowed_ids.append(m_id)
            else:
                # Cross family: check if knowledge category is portable
                if cat in ALLOWED_SELECTIVE_CATEGORIES:
                    allowed_memories.append(m)
                    allowed_ids.append(m_id)
                else:
                    filtered_ids.append(m_id)

        decision = TransferDecisionRecord(
            source_task_id=source_profile.task_id,
            target_task_id=target_profile.task_id,
            source_family=source_profile.task_family,
            target_family=target_profile.task_family,
            prior_affinity=prior_aff,
            historical_transfer_effect=hist_effect,
            evidence_volume=ev_vol,
            transfer_score=score,
            guard_mode=self.mode,
            action_taken="FILTER_TYPED",
            allowed_memory_ids=allowed_ids,
            filtered_memory_ids=filtered_ids,
            rationale=(
                f"SELECTIVE typed gating: {len(allowed_ids)} portable memories allowed, "
                f"{len(filtered_ids)} task-specific memories filtered across family boundary."
            ),
        )
        self._decision_history.append(decision)
        return allowed_memories, decision
