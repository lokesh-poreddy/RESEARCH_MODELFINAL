"""Focused RF-FLEX Phase 13.7 pilot checks."""
import json

from researchforge.benchmarks.rf_flex import RFFlexPilot, validate_rf_flex_artifacts
from researchforge.benchmarks.continuity.tasks import get_pilot_task_sequence


def test_rf_flex_pilot_emits_reconstructable_lineage(tmp_path):
    tasks = get_pilot_task_sequence(seed=0)[:2]
    summary = RFFlexPilot(
        tasks=tasks,
        conditions=["FULL_RESEARCHFORGE", "TRANSFER_GUARD_SELECTIVE"],
        n_generations=1,
        population_size=2,
    ).run(tmp_path)

    assert summary["run_count"] == 4
    assert summary["transfer_decision_count"] == 2
    assert (tmp_path / "experimental_protocol.json").exists()
    assert (tmp_path / "raw_trajectory_data.jsonl").exists()
    assert (tmp_path / "transfer_decisions.jsonl").exists()

    trajectory_rows = [json.loads(line) for line in (tmp_path / "raw_trajectory_data.jsonl").read_text().splitlines()]
    decision_rows = [json.loads(line) for line in (tmp_path / "decision_log.jsonl").read_text().splitlines()]
    transfer_rows = [json.loads(line) for line in (tmp_path / "transfer_decisions.jsonl").read_text().splitlines()]

    assert len(trajectory_rows) == 8
    assert len(decision_rows) == len(trajectory_rows)
    assert all(row["genome_id"] for row in trajectory_rows)
    assert all(row["outcome_id"] and row["diagnosis_id"] for row in trajectory_rows)
    assert {row["action_taken"] for row in transfer_rows} <= {"ALLOW_ALL", "FILTER_TYPED", "SUPPRESS_ALL"}
    assert all(row["provenance_id"] for row in transfer_rows)
    validation = json.loads((tmp_path / "trajectory_validation.json").read_text())
    assert validation["status"] == "PASS"
    assert validation["record_count"] == len(trajectory_rows)


def test_rf_flex_protocol_marks_pilot_as_non_confirmatory(tmp_path):
    RFFlexPilot(tasks=get_pilot_task_sequence(seed=0)[:1], n_generations=1, population_size=2).run(tmp_path)
    protocol = json.loads((tmp_path / "experimental_protocol.json").read_text())
    manifest = json.loads((tmp_path / "reproducibility_manifest.json").read_text())

    assert protocol["status"] == "FROZEN_PILOT_PROTOCOL"
    assert protocol["confirmatory_claims_permitted"] is False
    assert protocol["phase12b_artifacts_untouched"] is True
    assert manifest["artifact_count"] == 21


def test_rf_flex_replay_fingerprint_is_stable(tmp_path):
    pilot = RFFlexPilot(
        tasks=get_pilot_task_sequence(seed=0)[:2],
        conditions=["FULL_RESEARCHFORGE"],
        n_generations=1,
        population_size=2,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    pilot.run(first)
    pilot.run(second)

    first_validation = validate_rf_flex_artifacts(first)
    second_validation = validate_rf_flex_artifacts(second)
    assert first_validation["status"] == "PASS"
    assert second_validation["status"] == "PASS"
    assert first_validation["semantic_replay_fingerprint"] == second_validation["semantic_replay_fingerprint"]