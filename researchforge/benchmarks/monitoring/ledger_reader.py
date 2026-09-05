"""researchforge/benchmarks/monitoring/ledger_reader.py — Read-only streaming ledger reader.

Phase 13 (RF-1.0.0-beta.1):
Enforces Invariant 7: The monitor is strictly read-only with respect to Phase 12B.
Tails the append-only ledger using byte offsets without file locking or process contention.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..execution.models import LedgerEntry

logger = logging.getLogger(__name__)


class ExecutionLedgerStreamReader:
    """Non-blocking, read-only streaming consumer for phase12b_execution_ledger.jsonl."""

    def __init__(self, ledger_path: Path | str = "phase12b_execution_ledger.jsonl") -> None:
        self.ledger_path = Path(ledger_path)
        self._last_byte_offset: int = 0
        self._entries: List[LedgerEntry] = []

    @property
    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    @property
    def current_offset(self) -> int:
        return self._last_byte_offset

    def read_new_entries(self) -> List[LedgerEntry]:
        """Read newly appended entries since the last check.
        
        Uses file-pointer offset seeking; completely read-only and non-intrusive.
        """
        if not self.ledger_path.exists():
            return []

        new_entries: List[LedgerEntry] = []
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                f.seek(self._last_byte_offset)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    # Update offset after each full line
                    self._last_byte_offset = f.tell()
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record_dict = json.loads(line_str)
                        entry = LedgerEntry(**record_dict)
                        new_entries.append(entry)
                        self._entries.append(entry)
                    except Exception as e:
                        logger.warning("Failed to parse ledger line: %s (%s)", line_str[:60], e)
        except Exception as e:
            logger.error("Error reading execution ledger stream: %s", e)

        return new_entries

    def read_all_entries(self) -> List[LedgerEntry]:
        """Reset offset and read all entries from the beginning."""
        self._last_byte_offset = 0
        self._entries = []
        return self.read_new_entries()
