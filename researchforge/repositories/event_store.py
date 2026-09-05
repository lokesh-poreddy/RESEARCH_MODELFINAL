"""researchforge/repositories/event_store.py — Append-only research event store.

RF-1.0.0-alpha.3 (Phase 8A): Authoritative immutable research-history journal.
Enforces strictly monotonic sequence ordering, deterministic fingerprints, and non-mutability.
"""
from __future__ import annotations

import hashlib
import json
import time
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..domain.base import DomainObject


class EventStoreMutationError(Exception):
    """Raised when an attempt is made to mutate, delete, or overwrite a historical event."""
    pass


class EventOrderingError(Exception):
    """Raised when an event sequence number violates monotonicity for an aggregate or creates conflict."""
    pass


@dataclass(frozen=True)
class ResearchEventRecord(DomainObject):
    """Canonical stored event record in the append-only research journal."""
    event_type: str
    aggregate_id: str
    aggregate_type: str
    sequence_number: int
    payload: Dict[str, Any]
    provenance_id: str | None = None
    created_at: float | None = None
    global_sequence: int | None = None

    @property
    def event_id(self) -> str:
        return self.id

    @property
    def aggregate_sequence(self) -> int:
        return self.sequence_number

    @property
    def deterministic_fingerprint(self) -> str:
        return self.fingerprint()

    def canonical_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.id,
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "sequence_number": self.sequence_number,
            "global_sequence": self.global_sequence,
            "payload": self.payload,
            "provenance_id": self.provenance_id,
        }

    def fingerprint(self) -> str:
        content = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["event_id"] = self.id
        base["aggregate_sequence"] = self.sequence_number
        base["global_sequence"] = self.global_sequence
        base["deterministic_fingerprint"] = self.fingerprint()
        return base

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ResearchEventRecord":
        data = dict(obj)
        if "id" not in data and "event_id" in data:
            data["id"] = data.pop("event_id")
        else:
            data.pop("event_id", None)
        if "sequence_number" not in data and "aggregate_sequence" in data:
            data["sequence_number"] = data.pop("aggregate_sequence")
        else:
            data.pop("aggregate_sequence", None)
        data.pop("deterministic_fingerprint", None)
        return cls(**data)


class InMemoryEventStore:
    """In-memory reference implementation of the append-only event store."""

    def __init__(self) -> None:
        self._events_by_id: Dict[str, ResearchEventRecord] = {}
        self._events_by_aggregate: Dict[str, List[ResearchEventRecord]] = {}
        self._timeline: List[ResearchEventRecord] = []

    def append(self, event: ResearchEventRecord) -> None:
        if event.id in self._events_by_id:
            raise EventStoreMutationError(f"Cannot overwrite historical event with id '{event.id}'.")

        agg_events = self._events_by_aggregate.setdefault(event.aggregate_id, [])
        # Strict concurrency/ordering check: reject duplicate sequence for same aggregate
        for existing in agg_events:
            if existing.sequence_number == event.sequence_number:
                raise EventOrderingError(
                    f"Concurrency conflict on aggregate '{event.aggregate_id}': "
                    f"sequence {event.sequence_number} is already committed."
                )

        if agg_events and event.sequence_number <= agg_events[-1].sequence_number:
            raise EventOrderingError(
                f"Ordering violation for aggregate '{event.aggregate_id}': "
                f"attempted to append sequence {event.sequence_number} after {agg_events[-1].sequence_number}."
            )

        stored_event = event
        if stored_event.global_sequence is None:
            stored_event = dataclasses.replace(event, global_sequence=len(self._timeline) + 1)

        self._events_by_id[stored_event.id] = stored_event
        agg_events.append(stored_event)
        self._timeline.append(stored_event)

    def get_by_aggregate(self, aggregate_id: str) -> List[ResearchEventRecord]:
        return list(self._events_by_aggregate.get(aggregate_id, []))

    def get_all_events(self, since_sequence: int = 0) -> List[ResearchEventRecord]:
        return [e for e in self._timeline if (e.global_sequence or e.sequence_number) >= since_sequence]

    def get_by_id(self, event_id: str) -> Optional[ResearchEventRecord]:
        return self._events_by_id.get(event_id)

    def __len__(self) -> int:
        return len(self._timeline)
