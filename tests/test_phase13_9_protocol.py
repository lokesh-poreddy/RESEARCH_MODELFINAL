"""Phase 13.9 confirmatory design contract tests."""
import json

from researchforge.benchmarks.rf_flex_confirmatory import (
    CONDITIONS,
    TASK_UNIVERSE,
    RFFlexConfirmatoryProtocol,
    calibrate_from_pilot,
)
from researchforge.benchmarks.rf_flex import RFFlexPilot
from researchforge.benchmarks.continuity.tasks import get_pilot_task_sequence


def test_phase13_9_protocol_is_frozen_but_not_authorized():
    protocol = RFFlexConfirmatoryProtocol()
    protocol.validate()
    payload = protocol.to_dict()

    assert tuple(payload["conditions"]) == CONDITIONS
    assert tuple(payload["task_universe"]) == TASK_UNIVERSE
    assert payload["primary_endpoint"] == "decision_quality"
    assert payload["status"] == "DESIGN_ONLY"
    assert payload["confirmatory_execution_authorized"] is False
    assert protocol.fingerprint()


def test_phase13_9_calibration_is_descriptive_only(tmp_path):
    pilot_dir = tmp_path / "pilot"
    output_dir = tmp_path / "frozen"
    RFFlexPilot(
        tasks=get_pilot_task_sequence(seed=0)[:1],
        conditions=["FULL_RESEARCHFORGE"],
        n_generations=1,
        population_size=2,
    ).run(pilot_dir)

    protocol = RFFlexConfirmatoryProtocol()
    calibration = calibrate_from_pilot(pilot_dir, protocol)
    protocol.freeze(output_dir / "protocol.json")

    assert calibration["status"] == "CALIBRATION_ONLY"
    assert calibration["effect_size_or_power_claim"] is False
    assert calibration["confirmatory_groups"] == 8 * 3 * 5 * 6
    assert json.loads((output_dir / "protocol.json").read_text())["protocol_fingerprint"] == protocol.fingerprint()