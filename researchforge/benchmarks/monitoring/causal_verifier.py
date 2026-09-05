"""researchforge/benchmarks/monitoring/causal_verifier.py — On-the-fly DAG causal verifier.

Phase 13 (RF-1.0.0-beta.1):
Enforces DAG predecessor satisfaction and monotonic sequence order as
benchmark ledger records stream in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..execution.models import LedgerEntry


class LiveCausalVerifier:
    """Read-only verifier that checks incoming ledger entries against the frozen execution manifest."""

    def __init__(self, manifest_source: Path | str | Dict[str, Any] = "phase12b_execution_manifest.json") -> None:
        if isinstance(manifest_source, dict):
            self._manifest_data = manifest_source
        else:
            p = Path(manifest_source)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    self._manifest_data = json.load(f)
            else:
                self._manifest_data = {"entries": []}

        # Index manifest entries by entry_id
        self._manifest_map: Dict[str, Dict[str, Any]] = {}
        for entry in self._manifest_data.get("entries", []):
            eid = entry.get("entry_id")
            if eid:
                self._manifest_map[eid] = entry

        self._verified_entry_ids: Set[str] = set()
        self._violations: List[str] = []

    @property
    def verified_entry_ids(self) -> Set[str]:
        return set(self._verified_entry_ids)

    @property
    def violations(self) -> List[str]:
        return list(self._violations)

    def verify_entry(self, entry: LedgerEntry) -> Tuple[bool, Optional[str]]:
        """Verify an individual ledger entry against the manifest DAG.
        
        Returns:
            (is_valid, violation_message)
        """
        # 1. Check entry exists in manifest
        m_entry = self._manifest_map.get(entry.entry_id)
        if not m_entry:
            msg = f"Unknown entry_id '{entry.entry_id}' not found in frozen manifest."
            self._violations.append(msg)
            return False, msg

        # 2. Check field parity
        if m_entry.get("task_id") != entry.task_id:
            msg = f"Task ID mismatch for {entry.entry_id}: manifest={m_entry.get('task_id')}, ledger={entry.task_id}"
            self._violations.append(msg)
            return False, msg

        if m_entry.get("condition") != entry.condition:
            msg = f"Condition mismatch for {entry.entry_id}: manifest={m_entry.get('condition')}, ledger={entry.condition}"
            self._violations.append(msg)
            return False, msg

        if m_entry.get("sequence_position") != entry.sequence_position:
            msg = f"Sequence position mismatch for {entry.entry_id}: manifest={m_entry.get('sequence_position')}, ledger={entry.sequence_position}"
            self._violations.append(msg)
            return False, msg

        # 3. Check causal predecessor dependencies
        dependencies = m_entry.get("dependencies", [])
        for dep_id in dependencies:
            if dep_id not in self._verified_entry_ids:
                msg = (
                    f"Causal DAG violation for {entry.entry_id}: required predecessor "
                    f"'{dep_id}' has not been executed yet."
                )
                self._violations.append(msg)
                return False, msg

        self._verified_entry_ids.add(entry.entry_id)
        return True, None
