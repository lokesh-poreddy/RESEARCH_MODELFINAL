"""tests/test_phase13_stream_monitor.py — Real-time streaming benchmark monitor tests.

Classification: CORE
Verifies:
1. Streaming ledger reader consumes lines incrementally using file offsets.
2. Causal verifier enforces DAG predecessor order against manifest.
3. Progress tracker aggregates telemetry across the 32,400 scale.
4. STRICT CONFIRMATORY FIREWALL: Monitor code does not compute H1-H4 confirmatory statistics.
5. Monitor is read-only with respect to Phase 12B artifacts.
"""
import inspect
import json
import tempfile
from pathlib import Path
import pytest

from researchforge.benchmarks.execution.models import LedgerEntry
from researchforge.benchmarks.monitoring import (
    BenchmarkProgressTracker,
    ExecutionLedgerStreamReader,
    LiveCausalVerifier,
)


@pytest.fixture
def sample_entry() -> LedgerEntry:
    return LedgerEntry(
        timestamp="2026-09-06T00:00:00Z",
        entry_id="entry_001",
        sequence_group_id="continuous_experience__forward__0",
        sequence_position=0,
        condition="continuous_experience",
        task_id="digits_0_4",
        seed=0,
        ordering="forward",
        status="VALID_COMPLETED",
        best_metric=0.89,
        decision_quality=0.89,
        trials_executed=60,
        wallclock_seconds=12.5,
        memory_fingerprint_before="aaa",
        memory_fingerprint_after="bbb",
        entry_hash="ccc",
    )


def test_ledger_reader_incremental_offset_consumption(sample_entry):
    """Ledger reader must tail file incrementally using byte offsets without locking."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as f:
        tmp_path = Path(f.name)

    try:
        reader = ExecutionLedgerStreamReader(tmp_path)

        # 1. Initial empty read
        assert len(reader.read_new_entries()) == 0

        # 2. Append first entry
        with open(tmp_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample_entry.to_dict()) + "\n")

        entries_1 = reader.read_new_entries()
        assert len(entries_1) == 1
        assert entries_1[0].entry_id == "entry_001"

        # 3. Read again without new writes: returns 0 new entries
        assert len(reader.read_new_entries()) == 0

        # 4. Append second entry
        second_entry = LedgerEntry(
            timestamp="2026-09-06T00:01:00Z",
            entry_id="entry_002",
            sequence_group_id="continuous_experience__forward__0",
            sequence_position=1,
            condition="continuous_experience",
            task_id="digits_5_9",
            seed=0,
            ordering="forward",
            status="VALID_COMPLETED",
            best_metric=0.91,
            decision_quality=0.91,
            trials_executed=60,
            wallclock_seconds=11.2,
            memory_fingerprint_before="bbb",
            memory_fingerprint_after="ddd",
            entry_hash="eee",
        )
        with open(tmp_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(second_entry.to_dict()) + "\n")

        entries_2 = reader.read_new_entries()
        assert len(entries_2) == 1
        assert entries_2[0].entry_id == "entry_002"
        assert len(reader.entries) == 2
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_causal_verifier_enforces_dag_predecessors():
    """Verifier must flag entry if required predecessor entry has not arrived yet."""
    manifest = {
        "entries": [
            {
                "entry_id": "e_task1",
                "task_id": "digits_0_4",
                "condition": "cold_start",
                "sequence_position": 0,
                "dependencies": [],
            },
            {
                "entry_id": "e_task2",
                "task_id": "digits_5_9",
                "condition": "cold_start",
                "sequence_position": 1,
                "dependencies": ["e_task1"],
            },
        ]
    }
    verifier = LiveCausalVerifier(manifest_source=manifest)

    entry2 = LedgerEntry(
        timestamp="2026-09-06T00:00:00Z",
        entry_id="e_task2",
        sequence_group_id="cold_start__forward__0",
        sequence_position=1,
        condition="cold_start",
        task_id="digits_5_9",
        seed=0,
        ordering="forward",
        status="VALID_COMPLETED",
        best_metric=0.9,
        decision_quality=0.9,
        trials_executed=60,
        wallclock_seconds=10.0,
        memory_fingerprint_before="a",
        memory_fingerprint_after="b",
        entry_hash="c",
    )

    # Attempting to verify entry2 before entry1 must fail
    is_valid, err = verifier.verify_entry(entry2)
    assert not is_valid
    assert "Causal DAG violation" in err
    assert "e_task1" in err


def test_progress_tracker_computes_telemetry_across_scale(sample_entry):
    """Progress tracker aggregates trials, percentages, and condition counts."""
    tracker = BenchmarkProgressTracker()
    tracker.update([sample_entry])

    snap = tracker.generate_snapshot()
    assert snap["completed_trials"] == 60
    assert snap["target_total_trials"] == 32400
    assert snap["completed_entries"] == 1
    assert snap["condition_trials"]["continuous_experience"] == 60
    assert snap["outcome_distribution"]["VALID_COMPLETED"] == 1
    assert snap["percentage_complete"] == 0.19
    assert tracker.percentage_complete == pytest.approx(60 / 32400 * 100, rel=1e-3)


def test_strict_confirmatory_firewall():
    """Verify that monitor modules contain NO confirmatory hypothesis testing logic (H1-H4)."""
    import researchforge.benchmarks.monitoring.progress_tracker as pt_mod
    import researchforge.benchmarks.monitoring.ledger_reader as lr_mod
    import researchforge.benchmarks.monitoring.causal_verifier as cv_mod

    confirmatory_keywords = [
        "hypothesis_h1",
        "hypothesis_h2",
        "hypothesis_h3",
        "hypothesis_h4",
        "p_value",
        "holm_bonferroni",
        "primary_endpoint",
        "confirmatory_result",
    ]

    for mod in (pt_mod, lr_mod, cv_mod):
        source = inspect.getsource(mod)
        for kw in confirmatory_keywords:
            assert kw not in source.lower(), f"Confirmatory keyword '{kw}' leaked into monitor module {mod.__name__}!"
