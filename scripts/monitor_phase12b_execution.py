#!/usr/bin/env python3
"""scripts/monitor_phase12b_execution.py — CLI Monitor Daemon for Phase 12B Large-Scale Benchmark.

Phase 13 (RF-1.0.0-beta.1):
Tails phase12b_execution_ledger.jsonl in real time, validates DAG causality,
tracks completion %, throughput, ETA, and exports live observational snapshots.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchforge.benchmarks.monitoring import (
    BenchmarkProgressTracker,
    ExecutionLedgerStreamReader,
    LiveCausalVerifier,
)


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def render_dashboard(tracker: BenchmarkProgressTracker) -> None:
    snap = tracker.generate_snapshot()
    pct = snap["percentage_complete"]
    bar_width = 30
    filled = int((pct / 100.0) * bar_width)
    bar = "=" * filled + "-" * (bar_width - filled)

    print("\n" + "=" * 70)
    print(f" ResearchForge Phase 12B Large-Scale Benchmark Monitor [READ-ONLY]")
    print("=" * 70)
    print(f" Progress:       [{bar}] {pct:5.1f}%")
    print(f" Trials:         {snap['completed_trials']:,} / {snap['target_total_trials']:,}")
    print(f" Task Entries:   {snap['completed_entries']} / {snap['target_total_entries']}")
    print(f" Sequence Groups:{snap['completed_groups']} / {snap['target_total_groups']}")
    print(f" Elapsed:        {format_seconds(snap['elapsed_seconds'])}")
    print(f" Throughput:     {snap['trials_per_minute']:.1f} trials/min")
    print(f" Projected ETA:  {format_seconds(snap['estimated_seconds_remaining'])}")
    print("-" * 70)
    print(" Conditions Completed Trials:")
    for cond, count in snap["condition_trials"].items():
        print(f"   - {cond:22s}: {count:5,d} / 5,400 ({count/54.0:4.1f}%)")
    print("-" * 70)
    print(" Outcome Categories:")
    for out, count in snap["outcome_distribution"].items():
        print(f"   - {out:22s}: {count}")
    print("-" * 70)
    v_count = snap["causal_violations_count"]
    status_str = "OK (0 violations)" if v_count == 0 else f"VIOLATION DETECTED ({v_count})"
    print(f" Causal DAG Integrity: {status_str}")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time monitor for Phase 12B execution ledger")
    parser.add_argument("--ledger", type=str, default="phase12b_execution_ledger.jsonl", help="Ledger path")
    parser.add_argument("--manifest", type=str, default="phase12b_execution_manifest.json", help="Manifest path")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print status once and exit")
    parser.add_argument("--snapshot-file", type=str, default="phase12b_monitoring_snapshot.json", help="Snapshot output path")
    args = parser.parse_args()

    reader = ExecutionLedgerStreamReader(args.ledger)
    verifier = LiveCausalVerifier(args.manifest)
    tracker = BenchmarkProgressTracker(verifier)

    # Initial read
    entries = reader.read_new_entries()
    tracker.update(entries)
    tracker.export_snapshot(args.snapshot_file)
    render_dashboard(tracker)

    if args.once:
        return

    print("Monitoring stream... Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(args.interval)
            new_entries = reader.read_new_entries()
            if new_entries:
                tracker.update(new_entries)
                tracker.export_snapshot(args.snapshot_file)
                render_dashboard(tracker)
            if tracker.percentage_complete >= 100.0:
                print("Benchmark execution complete (100%).")
                break
    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")


if __name__ == "__main__":
    main()
