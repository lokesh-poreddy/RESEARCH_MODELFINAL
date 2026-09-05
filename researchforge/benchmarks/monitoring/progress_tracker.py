"""researchforge/benchmarks/monitoring/progress_tracker.py — Live benchmark progress tracker.

Phase 13 (RF-1.0.0-beta.1):
Enforces Criterion 9: STRICT CONFIRMATORY FIREWALL.
Computes runtime telemetry, throughput, ETA, and failure-distribution tracking.
Strictly prohibited from computing any H1–H4 confirmatory hypothesis statistics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..execution.models import LedgerEntry
from .causal_verifier import LiveCausalVerifier


class BenchmarkProgressTracker:
    """Aggregates observational runtime telemetry during 32,400-trial benchmark execution."""

    TARGET_TOTAL_TRIALS = 32400
    TARGET_TOTAL_ENTRIES = 540
    TARGET_TOTAL_GROUPS = 90
    TRIALS_PER_ENTRY = 60
    ENTRIES_PER_GROUP = 6

    def __init__(self, causal_verifier: Optional[LiveCausalVerifier] = None) -> None:
        self.causal_verifier = causal_verifier or LiveCausalVerifier()
        self.start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None

        self.completed_trials: int = 0
        self.completed_entries: int = 0
        self.condition_trials: Dict[str, int] = {
            "cold_start": 0,
            "no_memory": 0,
            "flat_ecrm": 0,
            "trajectory_memory": 0,
            "adaptive_trajectory": 0,
            "continuous_experience": 0,
        }
        self.outcome_distribution: Dict[str, int] = {}
        self.group_progress: Dict[str, int] = {}  # group_id -> count of completed entries
        self.completed_groups: int = 0
        self.recorded_entries: List[LedgerEntry] = []

    def update(self, entries: List[LedgerEntry]) -> None:
        """Process a batch of new ledger entries."""
        if not entries:
            return

        now = time.time()
        if self.start_time is None:
            self.start_time = now
        self.last_update_time = now

        for entry in entries:
            # Verify DAG causality
            self.causal_verifier.verify_entry(entry)

            # Update trial and entry counts
            t_count = entry.trials_executed or self.TRIALS_PER_ENTRY
            self.completed_trials += t_count
            self.completed_entries += 1

            # Condition breakdown
            c = entry.condition
            self.condition_trials[c] = self.condition_trials.get(c, 0) + t_count

            # Outcome category breakdown
            st = entry.status
            self.outcome_distribution[st] = self.outcome_distribution.get(st, 0) + 1

            # Group progression
            gid = entry.sequence_group_id
            self.group_progress[gid] = self.group_progress.get(gid, 0) + 1
            if self.group_progress[gid] == self.ENTRIES_PER_GROUP:
                self.completed_groups += 1

            self.recorded_entries.append(entry)

    @property
    def percentage_complete(self) -> float:
        return min(100.0, (self.completed_trials / self.TARGET_TOTAL_TRIALS) * 100.0)

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None or self.last_update_time is None:
            return 0.0
        return max(0.0, self.last_update_time - self.start_time)

    @property
    def trials_per_minute(self) -> float:
        el = self.elapsed_seconds
        if el <= 0.0:
            return 0.0
        return (self.completed_trials / el) * 60.0

    @property
    def estimated_seconds_remaining(self) -> Optional[float]:
        tpm = self.trials_per_minute
        if tpm <= 0.0:
            return None
        remaining_trials = max(0, self.TARGET_TOTAL_TRIALS - self.completed_trials)
        return (remaining_trials / tpm) * 60.0

    def generate_snapshot(self) -> Dict[str, Any]:
        """Generate machine-readable telemetry snapshot.
        
        Strict Invariant: Zero H1–H4 confirmatory statistics are contained herein.
        """
        return {
            "timestamp": time.time(),
            "target_total_trials": self.TARGET_TOTAL_TRIALS,
            "completed_trials": self.completed_trials,
            "percentage_complete": round(self.percentage_complete, 2),
            "target_total_entries": self.TARGET_TOTAL_ENTRIES,
            "completed_entries": self.completed_entries,
            "target_total_groups": self.TARGET_TOTAL_GROUPS,
            "completed_groups": self.completed_groups,
            "condition_trials": dict(self.condition_trials),
            "outcome_distribution": dict(self.outcome_distribution),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "trials_per_minute": round(self.trials_per_minute, 1),
            "estimated_seconds_remaining": round(self.estimated_seconds_remaining, 1) if self.estimated_seconds_remaining is not None else None,
            "causal_violations_count": len(self.causal_verifier.violations),
            "causal_violations": list(self.causal_verifier.violations[:10]),
        }

    def export_snapshot(self, output_path: Path | str = "phase12b_monitoring_snapshot.json") -> Path:
        """Export snapshot to disk (for dashboard/monitoring tools)."""
        p = Path(output_path)
        data = self.generate_snapshot()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return p
