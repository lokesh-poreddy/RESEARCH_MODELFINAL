"""Phase 13.9 confirmatory RF-FLEX evaluation contract.

The contract is intentionally separate from Phase 12B artifacts and does not
execute the confirmatory benchmark. Pilot telemetry may calibrate runtime and
storage estimates, but never changes the frozen task, condition, or endpoint
definitions.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


TASK_UNIVERSE = (
    "digits_0_4",
    "digits_5_9",
    "digits_all_10",
    "synthetic_ecg_lead1",
    "synthetic_ecg_lead2",
    "xor_parity_8bit",
)

CONDITIONS = (
    "COLD_START",
    "NO_MEMORY",
    "TRAJECTORY_MEMORY",
    "ADAPTIVE_TRAJECTORY",
    "TRANSFER_GUARD_STRICT",
    "TRANSFER_GUARD_SELECTIVE",
    "DYNAMIC_ROUTER",
    "FULL_RESEARCHFORGE",
)


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RFFlexConfirmatoryProtocol:
    """Frozen design contract for Phase 13.9."""

    protocol_version: str = "13.9.0"
    status: str = "DESIGN_ONLY"
    task_universe: Sequence[str] = TASK_UNIVERSE
    conditions: Sequence[str] = CONDITIONS
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    orderings: Sequence[str] = ("forward", "reverse", "cross_first")
    generations_per_task: int = 10
    population_size: int = 6
    primary_endpoint: str = "decision_quality"
    secondary_endpoints: Sequence[str] = (
        "best_metric",
        "research_efficiency",
        "experiments_to_best",
        "successful_decisions_rate",
        "redundant_experiment_rate",
        "time_to_best",
        "resource_cost_to_best",
        "experience_reuse_rate",
        "useful_transfer_rate",
        "harmful_transfer_rate",
        "memory_retrieval_precision",
        "decision_change_due_to_experience",
        "repeated_failure_rate",
        "mode_transition_accuracy",
    )
    transfer_regimes: Sequence[str] = ("SAME_FAMILY", "CROSS_FAMILY", "UNRELATED")
    routing_regimes: Sequence[str] = (
        "FIXED_DEEP_RESEARCH",
        "FIXED_EXPERIMENTAL",
        "FIXED_REPLICATION",
        "DYNAMIC_ROUTER",
    )
    statistical_method: Mapping[str, Any] = field(default_factory=lambda: {
        "primary_test": "paired_permutation_test",
        "pairing_unit": "task_ordering_seed_block",
        "primary_alpha": 0.05,
        "secondary_multiplicity": "holm_bonferroni",
        "effect_size": "paired_cohens_dz",
        "confidence_interval": "bootstrap_95_percent",
        "no_outlier_removal": True,
    })
    exclusion_rules: Sequence[str] = (
        "no_post_hoc_task_removal",
        "no_post_hoc_seed_removal",
        "no_post_hoc_ordering_removal",
        "no_outlier_deletion",
        "incomplete_trajectory_invalid",
        "invalid_execution_retained_and_endpoint_excluded_only_by_rule",
    )
    stopping_rules: Sequence[str] = (
        "stop_on_fatal_protocol_integrity_failure",
        "stop_on_future_information_leakage",
        "stop_on_artifact_hash_mismatch",
        "otherwise_run_full_frozen_cohort",
    )
    phase12b_boundary: str = "read_only; no Phase 12B artifact mutation"
    confirmatory_execution_authorized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "status": self.status,
            "task_universe": list(self.task_universe),
            "conditions": list(self.conditions),
            "seeds": list(self.seeds),
            "orderings": list(self.orderings),
            "generations_per_task": self.generations_per_task,
            "population_size": self.population_size,
            "primary_endpoint": self.primary_endpoint,
            "secondary_endpoints": list(self.secondary_endpoints),
            "transfer_regimes": list(self.transfer_regimes),
            "routing_regimes": list(self.routing_regimes),
            "statistical_method": dict(self.statistical_method),
            "exclusion_rules": list(self.exclusion_rules),
            "stopping_rules": list(self.stopping_rules),
            "phase12b_boundary": self.phase12b_boundary,
            "confirmatory_execution_authorized": self.confirmatory_execution_authorized,
        }

    def validate(self) -> None:
        payload = self.to_dict()
        if self.status != "DESIGN_ONLY":
            raise ValueError("Phase 13.9 protocol must remain DESIGN_ONLY until explicitly authorized")
        if tuple(self.conditions) != CONDITIONS:
            raise ValueError("Phase 13.9 condition order/content is frozen")
        if tuple(self.task_universe) != TASK_UNIVERSE:
            raise ValueError("Phase 13.9 task universe is frozen")
        if self.primary_endpoint != "decision_quality":
            raise ValueError("decision_quality is the frozen primary endpoint")
        if self.confirmatory_execution_authorized:
            raise ValueError("Confirmatory execution cannot be authorized by the design artifact")
        if self.generations_per_task <= 0 or self.population_size <= 0 or not self.seeds:
            raise ValueError("Protocol budget and seed policy must be non-empty")
        if len(set(payload["secondary_endpoints"])) != len(payload["secondary_endpoints"]):
            raise ValueError("Secondary endpoints must be unique")

    def fingerprint(self) -> str:
        self.validate()
        return _fingerprint(self.to_dict())

    def freeze(self, output_path: Path | str) -> Dict[str, Any]:
        self.validate()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            **self.to_dict(),
            "protocol_fingerprint": self.fingerprint(),
            "python": sys.version,
            "platform": platform.platform(),
        }
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifact


def calibrate_from_pilot(pilot_directory: Path | str, protocol: RFFlexConfirmatoryProtocol | None = None) -> Dict[str, Any]:
    """Estimate confirmatory resource needs from pilot telemetry only."""
    protocol = protocol or RFFlexConfirmatoryProtocol()
    protocol.validate()
    pilot_path = Path(pilot_directory)
    validation_path = pilot_path / "trajectory_validation.json"
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing Phase 13.8 validation artifact: {validation_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise ValueError("Cannot calibrate from a failed Phase 13.8 pilot")
    pilot_groups = int(validation["observed_trajectory_groups"])
    pilot_records = int(validation["record_count"])
    pilot_runtime = float(validation.get("runtime_seconds") or 0.0)
    pilot_storage = int(validation.get("storage_bytes") or 0)
    confirmatory_groups = len(protocol.conditions) * len(protocol.orderings) * len(protocol.seeds) * len(protocol.task_universe)
    scale = confirmatory_groups / pilot_groups if pilot_groups else 0.0
    return {
        "calibration_version": "13.9.0",
        "status": "CALIBRATION_ONLY",
        "pilot_validation_fingerprint": validation.get("semantic_replay_fingerprint"),
        "pilot_groups": pilot_groups,
        "pilot_records": pilot_records,
        "pilot_runtime_seconds": pilot_runtime,
        "pilot_storage_bytes": pilot_storage,
        "confirmatory_groups": confirmatory_groups,
        "estimated_runtime_seconds": round(pilot_runtime * scale, 3),
        "estimated_storage_bytes": round(pilot_storage * scale),
        "scaling_assumption": "linear extrapolation from validated pilot telemetry",
        "effect_size_or_power_claim": False,
        "protocol_fingerprint": protocol.fingerprint(),
    }
