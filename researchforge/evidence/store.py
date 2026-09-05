"""Evidence Store: centralized repository for evidence, claims, contradictions,
and decision-time temporal snapshots.

Scientific Principles (Phase 8):
1. Semantic deduplication: Identical observations do not multiply indefinitely,
   while genuinely distinct observations are strictly preserved.
2. Contradiction preservation: Both supporting and contradicting evidence for a claim
   remain simultaneously accessible and are never silently averaged away.
3. Decision-time snapshots: Reconstructs exact evidence availability as of a specific
   timestamp to prevent temporal leakage in decision evaluation.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..domain.claim import Claim, ClaimStatus, EvidenceRelation
from ..domain.evidence import Evidence
from .snapshot import EvidenceSnapshot


class EvidenceStore:
    """In-memory canonical evidence repository with temporal snapshotting."""

    def __init__(self) -> None:
        self._evidence: Dict[str, Evidence] = {}
        self._claims: Dict[str, Claim] = {}
        self._seen_semantic_keys: Set[Tuple[Any, ...]] = set()
        # Chronological timeline: list of (timestamp, evidence_id)
        self._timeline: List[Tuple[float, str]] = []

    def store(self, evidence: Evidence) -> bool:
        """Store an evidence record. Returns True if newly added, False if duplicate."""
        if evidence.id in self._evidence:
            return False

        sem_key = evidence.semantic_key()
        if sem_key in self._seen_semantic_keys:
            return False

        self._evidence[evidence.id] = evidence
        self._seen_semantic_keys.add(sem_key)
        ts = evidence.created_at if evidence.created_at is not None else time.time()
        self._timeline.append((ts, evidence.id))
        return True

    add_evidence = store

    def __len__(self) -> int:
        return len(self._evidence)

    def register_claim(self, claim: Claim) -> bool:
        """Register a scientific claim."""
        if claim.id in self._claims:
            return False
        self._claims[claim.id] = claim
        return True

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self._claims.get(claim_id)

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        return self._evidence.get(evidence_id)

    def attach_evidence_to_claim(
        self,
        claim_id: str,
        evidence_id: str,
        relation: str | EvidenceRelation,
    ) -> Claim:
        """Attach evidence to a claim, updating the claim's status while preserving contradictions."""
        claim = self._claims.get(claim_id)
        if claim is None:
            raise KeyError(f"Claim with id '{claim_id}' not found.")
        evidence = self._evidence.get(evidence_id)
        if evidence is None:
            raise KeyError(f"Evidence with id '{evidence_id}' not found.")

        updated_claim = claim.add_evidence(evidence_id, relation)
        self._claims[claim_id] = updated_claim
        return updated_claim

    def get_claim_evidence(self, claim_id: str) -> Dict[str, List[Evidence]]:
        """Retrieve all structured evidence partitioned by relationship."""
        claim = self._claims.get(claim_id)
        if claim is None:
            raise KeyError(f"Claim with id '{claim_id}' not found.")

        return {
            "supported_by": [self._evidence[eid] for eid in (claim.supporting_evidence_ids or []) if eid in self._evidence],
            "contradicted_by": [self._evidence[eid] for eid in (claim.contradicting_evidence_ids or []) if eid in self._evidence],
            "uncertain_from": [self._evidence[eid] for eid in (claim.uncertain_evidence_ids or []) if eid in self._evidence],
            "speculative_from": [self._evidence[eid] for eid in (claim.speculative_evidence_ids or []) if eid in self._evidence],
        }

    def get_contradicting_evidence_for_claim(self, claim_id: str) -> List[Evidence]:
        """Retrieve all contradictory evidence items associated with a claim."""
        return self.get_claim_evidence(claim_id)["contradicted_by"]

    def create_decision_snapshot(
        self,
        decision_id: str,
        as_of_timestamp: float,
        provenance_id: Optional[str] = None,
    ) -> EvidenceSnapshot:
        """Create a deterministic, temporally isolated snapshot of available evidence as of timestamp."""
        eligible_ids = [
            eid for ts, eid in self._timeline
            if ts <= as_of_timestamp
        ]
        eligible_items = [
            self._evidence[eid] for eid in eligible_ids
            if eid in self._evidence
        ]
        return EvidenceSnapshot.create(
            decision_id=decision_id,
            as_of_timestamp=as_of_timestamp,
            evidence_ids=eligible_ids,
            provenance_id=provenance_id,
            evidence_items=eligible_items,
        )

    def get_evidence_available_at(self, as_of_timestamp: float) -> List[Evidence]:
        """Retrieve evidence items that were registered at or before as_of_timestamp."""
        return [
            self._evidence[eid]
            for ts, eid in self._timeline
            if ts <= as_of_timestamp and eid in self._evidence
        ]
