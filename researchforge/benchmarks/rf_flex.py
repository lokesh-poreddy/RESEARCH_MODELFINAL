"""RF-FLEX pilot harness for full ResearchForge trajectory evaluation.

This module is an execution and provenance harness, not a confirmatory
statistical engine. It keeps the existing continuity task definitions and
controller as the source of experiment behavior, then emits paper-ready pilot
streams for each task boundary and decision.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .continuity.tasks import ContinuityTask, get_ordered_tasks, get_pilot_task_sequence
from ..pipeline.controller import ResearchController, RunResult
from ..transfer.affinity import TaskProfile
from ..transfer.guard import TransferGuard
from ..transfer.types import TransferGuardMode
from ..router.router import DynamicModeRouter


CONDITION_CONFIG = {
    "COLD_START": {"memory": False, "guard": TransferGuardMode.STRICT},
    "FULL_RESEARCHFORGE": {"memory": True, "guard": TransferGuardMode.DISABLED},
    "TRANSFER_GUARD_SELECTIVE": {"memory": True, "guard": TransferGuardMode.SELECTIVE},
}


@dataclass
class RFFlexRun:
    condition: str
    ordering: str
    seed: int
    task_id: str
    task_index: int
    prior_task_count: int
    task_fingerprint: str
    memory_before: str
    memory_after: str
    result: RunResult
    problem_id: str
    rdg: Dict[str, Any]
    states: List[Dict[str, Any]]
    transfer_decision: Optional[Dict[str, Any]] = None
    mode_transitions: List[Dict[str, Any]] = field(default_factory=list)


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _fingerprint(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _profile(task: ContinuityTask) -> TaskProfile:
    classes = len(set(task.task.y_train.tolist()))
    modality = {
        "digits_vision": "spatial",
        "signal_processing": "temporal",
        "discrete_logic": "tabular",
    }.get(task.family, "unknown")
    return TaskProfile(
        task_id=task.task_id,
        task_family=task.family,
        modality=modality,
        n_features=int(task.task.X_train.shape[1]),
        n_classes=classes,
        is_synthetic="synthetic" in task.task.name or "xor" in task.task.name,
    )


def _memory_rows(ecrm: Any) -> List[Dict[str, Any]]:
    return [record.to_dict() for record in ecrm.records.values()]


def _filtered_ecrm(source_ecrm: Any, allowed_ids: set[str]) -> Any:
    from ..memory.ecrm import ECRM

    filtered = ECRM(
        decay_lambda=source_ecrm.decay_lambda,
        retention_threshold=source_ecrm.retention_threshold,
        promotion_threshold=source_ecrm.promotion_threshold,
    )
    for record in source_ecrm.records.values():
        if record.id in allowed_ids:
            filtered.store(
                text_summary=record.text_summary,
                context=record.context,
                outcome=record.outcome,
                strategy=record.strategy,
            )
    return filtered


class RFFlexPilot:
    """Runs a bounded multi-task pilot and writes reconstructable artifacts."""

    def __init__(
        self,
        tasks: Optional[List[ContinuityTask]] = None,
        conditions: Optional[List[str]] = None,
        orderings: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        n_generations: int = 3,
        population_size: int = 4,
    ) -> None:
        self.tasks = tasks or get_pilot_task_sequence(seed=0)
        self.conditions = conditions or list(CONDITION_CONFIG)
        self.orderings = orderings or ["forward"]
        self.seeds = seeds or [0]
        self.n_generations = n_generations
        self.population_size = population_size
        unknown = set(self.conditions) - set(CONDITION_CONFIG)
        if unknown:
            raise ValueError(f"Unknown RF-FLEX conditions: {sorted(unknown)}")

    def run(self, output_dir: Path | str) -> Dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        started = time.time()
        runs: List[RFFlexRun] = []
        transfer_rows: List[Dict[str, Any]] = []
        transition_rows: List[Dict[str, Any]] = []

        for condition in self.conditions:
            config = CONDITION_CONFIG[condition]
            for ordering in self.orderings:
                for seed in self.seeds:
                    inherited_ecrm = None
                    inherited_policy = None
                    inherited_fails = None
                    prior_task_count = 0
                    prior_family: Optional[str] = None
                    prior_task_id: Optional[str] = None
                    prior_profile: Optional[TaskProfile] = None
                    transfer_history: List[Dict[str, Any]] = []

                    for task_index, task in enumerate(get_ordered_tasks(self.tasks, ordering)):
                        router = DynamicModeRouter()
                        guard = TransferGuard(mode=config["guard"])
                        memory_before = "none"
                        decision = None
                        inherited_for_task = inherited_ecrm if config["memory"] else None

                        if inherited_for_task is not None:
                            memory_before = _fingerprint(_memory_rows(inherited_for_task))
                        if prior_profile is not None and inherited_for_task is not None:
                            current_profile = _profile(task)
                            memories = _memory_rows(inherited_for_task)
                            allowed, decision_record = guard.filter_memories(
                                memories,
                                prior_profile,
                                current_profile,
                                transfer_history=transfer_history,
                            )
                            decision = decision_record.to_dict()
                            decision.update({
                                "condition": condition,
                                "ordering": ordering,
                                "seed": seed,
                                "task_sequence_index": task_index,
                                "source_task_id": prior_task_id,
                            })
                            transfer_rows.append(decision)
                            if decision_record.action_taken != "ALLOW_ALL":
                                inherited_for_task = _filtered_ecrm(
                                    inherited_for_task,
                                    {row["id"] for row in allowed},
                                )
                            router.handle_transfer_context(
                                is_cross_family=prior_family != task.family,
                                prior_task_count=prior_task_count,
                            )

                        controller = ResearchController(
                            task.task,
                            condition="full" if config["memory"] else "cold_start",
                            seed=seed,
                            population_size=self.population_size,
                            ecrm=inherited_for_task,
                            policy_learner=inherited_policy if config["memory"] else None,
                            failed_signatures=inherited_fails if config["memory"] else None,
                            enable_dynamic_router=True,
                            router=router,
                            transfer_guard=guard,
                        )
                        result = controller.run(n_generations=self.n_generations)
                        memory_after = _fingerprint(_memory_rows(controller.ecrm))
                        transitions = [event.to_dict() for event in router.transition_history]
                        transition_rows.extend({
                            **event,
                            "condition": condition,
                            "ordering": ordering,
                            "seed": seed,
                            "task_id": task.task_id,
                            "task_sequence_index": task_index,
                        } for event in transitions)
                        runs.append(RFFlexRun(
                            condition=condition,
                            ordering=ordering,
                            seed=seed,
                            task_id=task.task_id,
                            task_index=task_index,
                            prior_task_count=prior_task_count,
                            task_fingerprint=task.fingerprint(),
                            memory_before=memory_before,
                            memory_after=memory_after,
                            result=result,
                            problem_id=controller.problem.id,
                            rdg=controller.rdg.to_dict(),
                            states=[state.to_dict() for state in result.states],
                            transfer_decision=decision,
                            mode_transitions=transitions,
                        ))

                        if config["memory"]:
                            inherited_ecrm = controller.ecrm
                            inherited_policy = controller.policy
                            inherited_fails = set(controller._failed_signatures)
                            prior_task_count += 1
                        prior_family = task.family
                        prior_task_id = task.task_id
                        prior_profile = _profile(task)

        self._write_artifacts(out, runs, transfer_rows, transition_rows, started)
        return {
            "output_dir": str(out),
            "run_count": len(runs),
            "transition_count": len(transition_rows),
            "transfer_decision_count": len(transfer_rows),
            "elapsed_seconds": time.time() - started,
        }

    def _write_artifacts(
        self,
        out: Path,
        runs: List[RFFlexRun],
        transfer_rows: List[Dict[str, Any]],
        transition_rows: List[Dict[str, Any]],
        started: float,
    ) -> None:
        protocol = {
            "benchmark": "RF-FLEX",
            "protocol_version": "13.8.0",
            "version": "0.1.0-pilot",
            "status": "FROZEN_PILOT_PROTOCOL",
            "conditions": self.conditions,
            "orderings": self.orderings,
            "seeds": self.seeds,
            "n_generations": self.n_generations,
            "population_size": self.population_size,
            "task_universe": [task.task_id for task in self.tasks],
            "seed_policy": "explicit_seed_list_per_condition_ordering",
            "resource_budget": {"generations": self.n_generations, "population_size": self.population_size},
            "termination_rules": ["generation_budget_exhausted", "controller_execution_failure"],
            "transfer_policy": {condition: CONDITION_CONFIG[condition]["guard"].value for condition in self.conditions},
            "router_policy": "DynamicModeRouter policy 1.0.0-beta.1",
            "recording_schema": "rf-flex-lineage-v1",
            "primary_endpoints": ["decision_quality", "best_metric"],
            "secondary_endpoints": ["research_efficiency", "experiments_to_best", "repeated_failure_rate", "transfer_action", "mode_transition_count"],
            "exclusion_rules": ["no_post_hoc_row_deletion", "incomplete_trajectory_is_invalid"],
            "confirmatory_claims_permitted": False,
            "phase12b_artifacts_untouched": True,
        }
        _write_json(out / "experimental_protocol.json", {**protocol, "fingerprint": _fingerprint(protocol)})
        _write_json(out / "condition_manifest.json", {"conditions": CONDITION_CONFIG})
        _write_json(out / "task_manifest.json", {"tasks": [task.to_dict() for task in self.tasks]})

        raw_rows = []
        decision_rows = []
        memory_rows = []
        state_rows = []
        genome_rows = []
        rdg_rows = []
        diagnosis_rows = []
        outcome_rows = []
        for run in runs:
            nodes = {node["id"]: node for node in run.rdg["nodes"]}
            problem = nodes[run.problem_id]
            rdg_rows.extend({
                "condition": run.condition,
                "ordering": run.ordering,
                "seed": run.seed,
                "task_id": run.task_id,
                "node": node,
            } for node in run.rdg["nodes"])
            ordered_trials = list(run.result.trials)
            trial_decision_ids = [
                self._decision_id(run, trial.generation, index)
                for index, trial in enumerate(ordered_trials)
            ]
            state_rows.extend({
                "condition": run.condition,
                "ordering": run.ordering,
                "seed": run.seed,
                "task_id": run.task_id,
                **state,
            } for state in run.states)
            for index, trial in enumerate(ordered_trials):
                generation = trial.generation
                hypotheses = [
                    node for node in nodes.values()
                    if node["type"] == "Hypothesis"
                    and node.get("attributes", {}).get("generation") == generation
                ]
                experiments = [
                    node for node in nodes.values()
                    if node["type"] == "Experiment"
                    and node.get("attributes", {}).get("genome_id") == trial.genome_id
                ]
                if len(hypotheses) != 1 or len(experiments) != 1:
                    raise ValueError(f"Incomplete RDG lineage for {run.task_id} generation {generation}")
                hypothesis = hypotheses[0]
                experiment = experiments[0]
                findings = [
                    node for node in nodes.values()
                    if node["type"] == "Finding"
                    and any(edge["from_id"] == experiment["id"] and edge["to_id"] == node["id"] for edge in run.rdg["edges"])
                ]
                if len(findings) != 1:
                    raise ValueError(f"Incomplete finding lineage for {experiment['id']}")
                state = next((state for state in run.states if state["generation"] == generation), None)
                if state is None:
                    raise ValueError(f"Missing ResearchState for {run.task_id} generation {generation}")
                decision_id = trial_decision_ids[index]
                previous_decision_id = trial_decision_ids[index - 1] if index else None
                next_decision_id = trial_decision_ids[index + 1] if index + 1 < len(trial_decision_ids) else None
                transfer_id = run.transfer_decision.get("provenance_id") if run.transfer_decision else None
                base = {
                    "condition": run.condition,
                    "ordering": run.ordering,
                    "seed": run.seed,
                    "task_id": run.task_id,
                    "task_sequence_index": run.task_index,
                    "prior_task_count": run.prior_task_count,
                    "generation": generation,
                    "problem_id": problem["id"],
                    "research_state_id": state["state_id"],
                    "hypothesis_id": hypothesis["id"],
                    "decision_id": decision_id,
                    "previous_decision_id": previous_decision_id,
                    "next_decision_id": next_decision_id,
                    "mode": run.mode_transitions[-1]["state_after"] if run.mode_transitions else "DEEP_RESEARCH",
                    "memory_context_ids": run.transfer_decision.get("allowed_memory_ids", []) if run.transfer_decision else [],
                    "transfer_provenance_id": transfer_id,
                    "genome_id": trial.genome_id,
                    "experiment_id": experiment["id"],
                    "validity_verdict": "VALID_COMPLETED" if trial.failure == "None" else "SCIENTIFIC_FAILURE",
                    "finding_id": findings[0]["id"],
                    "outcome_id": self._lineage_id(run, "outcome", generation, index),
                    "diagnosis_id": self._lineage_id(run, "diagnosis", generation, index),
                    "experience_id": self._lineage_id(run, "experience", generation, index),
                    "policy_version": "controller_policy",
                    "strategy": trial.strategy,
                    "model_type": trial.model_type,
                    "metric": trial.metric,
                    "best_so_far": trial.best_so_far,
                    "failure": trial.failure,
                }
                base["trajectory_fingerprint"] = _fingerprint({key: base[key] for key in (
                    "problem_id", "research_state_id", "hypothesis_id", "decision_id",
                    "genome_id", "experiment_id", "finding_id", "outcome_id",
                    "diagnosis_id", "experience_id", "generation",
                )})
                raw_rows.append(base)
                decision_rows.append({k: base[k] for k in ("decision_id", "task_id", "generation", "mode", "strategy", "memory_context_ids", "transfer_provenance_id", "policy_version", "previous_decision_id", "next_decision_id")})
                genome_rows.append({"genome_id": trial.genome_id, "task_id": run.task_id, "generation": trial.generation, "model_type": trial.model_type})
                diagnosis_rows.append({"diagnosis_id": base["diagnosis_id"], "failure": trial.failure, "task_id": run.task_id, "generation": trial.generation})
                outcome_rows.append({"outcome_id": base["outcome_id"], "experiment_id": base["experiment_id"], "metric": trial.metric, "validity_verdict": base["validity_verdict"]})
            memory_rows.append({"task_id": run.task_id, "condition": run.condition, "memory_before": run.memory_before, "memory_after": run.memory_after, "transfer_decision": run.transfer_decision})

        _write_jsonl(out / "raw_trajectory_data.jsonl", raw_rows)
        _write_jsonl(out / "decision_log.jsonl", decision_rows)
        _write_jsonl(out / "memory_events.jsonl", memory_rows)
        _write_jsonl(out / "research_states.jsonl", state_rows)
        _write_jsonl(out / "mode_transitions.jsonl", transition_rows)
        _write_jsonl(out / "transfer_decisions.jsonl", transfer_rows)
        _write_jsonl(out / "genome_lineage.jsonl", genome_rows)
        _write_jsonl(out / "rdg_nodes.jsonl", rdg_rows)
        _write_jsonl(out / "failure_diagnoses.jsonl", diagnosis_rows)
        _write_jsonl(out / "outcome_records.jsonl", outcome_rows)

        summaries = []
        for run in runs:
            curve = [trial.best_so_far for trial in run.result.trials]
            summaries.append({"condition": run.condition, "task_id": run.task_id, "seed": run.seed, "best_metric": run.result.best_metric, "experiments_to_best": next((i for i, value in enumerate(curve) if value == run.result.best_metric), len(curve)), "research_improvement_curve": curve, "auc_research": sum(curve)})
        _write_json(out / "statistical_analysis.json", {"status": "DESCRIPTIVE_PILOT", "n": len(summaries), "summaries": summaries})
        _write_json(out / "ablation_results.json", {"status": "DESCRIPTIVE_PILOT", "by_condition": {condition: [row for row in summaries if row["condition"] == condition] for condition in self.conditions}})
        _write_json(out / "transfer_analysis.json", {"status": "DESCRIPTIVE_PILOT", "decisions": transfer_rows})
        _write_json(out / "routing_analysis.json", {"status": "DESCRIPTIVE_PILOT", "transitions": transition_rows})
        _write_json(out / "failure_learning_analysis.json", {"status": "DESCRIPTIVE_PILOT", "repeated_failure_rate": self._repeated_failure_rate(runs)})
        _write_json(out / "trajectory_analysis.json", {"status": "DESCRIPTIVE_PILOT", "trajectories": summaries})
        manifest = {"benchmark": "RF-FLEX", "artifact_count": 21, "run_count": len(runs), "elapsed_seconds": time.time() - started, "python": sys.version, "platform": platform.platform()}
        _write_json(out / "reproducibility_manifest.json", {**manifest, "fingerprint": _fingerprint(manifest)})

        validation = validate_rf_flex_artifacts(out)
        _write_json(out / "trajectory_validation.json", validation)

    @staticmethod
    def _decision_id(run: RFFlexRun, generation: int, index: int) -> str:
        return f"decision_{run.condition}_{run.ordering}_{run.seed}_{run.task_id}_{generation}_{index}"

    @staticmethod
    def _lineage_id(run: RFFlexRun, kind: str, generation: int, index: int) -> str:
        return f"{kind}_{run.condition}_{run.ordering}_{run.seed}_{run.task_id}_{generation}_{index}"

    @staticmethod
    def _repeated_failure_rate(runs: List[RFFlexRun]) -> float:
        failures = 0
        repeated = 0
        for run in runs:
            seen = set()
            for trial in run.result.trials:
                if trial.failure == "None":
                    continue
                failures += 1
                signature = (trial.strategy, trial.model_type)
                if signature in seen:
                    repeated += 1
                seen.add(signature)
        return repeated / failures if failures else 0.0


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_rf_flex_artifacts(output_dir: Path | str) -> Dict[str, Any]:
    """Validate complete RF-FLEX lineage and fail closed on missing edges."""
    out = Path(output_dir)
    raw = _read_jsonl(out / "raw_trajectory_data.jsonl")
    states = _read_jsonl(out / "research_states.jsonl")
    rdg = _read_jsonl(out / "rdg_nodes.jsonl")
    transfer = _read_jsonl(out / "transfer_decisions.jsonl")
    protocol = json.loads((out / "experimental_protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    state_ids = {row["state_id"] for row in states}
    node_ids = {row["node"]["id"] for row in rdg}
    transfer_ids = {row["provenance_id"] for row in transfer}
    required = ("problem_id", "research_state_id", "hypothesis_id", "decision_id", "genome_id", "experiment_id", "finding_id", "outcome_id", "diagnosis_id", "experience_id", "trajectory_fingerprint")
    errors: List[str] = []
    ordered_groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in raw:
        missing = [key for key in required if not row.get(key)]
        if missing:
            errors.append(f"{row.get('task_id')} generation {row.get('generation')}: missing {missing}")
            continue
        if row["research_state_id"] not in state_ids:
            errors.append(f"{row['decision_id']}: state does not exist")
        for key in ("problem_id", "hypothesis_id", "genome_id", "experiment_id", "finding_id"):
            if row[key] not in node_ids and key != "genome_id":
                errors.append(f"{row['decision_id']}: {key} does not exist")
        if row["transfer_provenance_id"] and row["transfer_provenance_id"] not in transfer_ids:
            errors.append(f"{row['decision_id']}: transfer provenance does not exist")
        expected_fp = _fingerprint({key: row[key] for key in ("problem_id", "research_state_id", "hypothesis_id", "decision_id", "genome_id", "experiment_id", "finding_id", "outcome_id", "diagnosis_id", "experience_id", "generation")})
        if row["trajectory_fingerprint"] != expected_fp:
            errors.append(f"{row['decision_id']}: trajectory fingerprint mismatch")
        group = (row["condition"], row["ordering"], row["seed"], row["task_id"])
        ordered_groups.setdefault(group, []).append(row)

    for group, rows in ordered_groups.items():
        rows.sort(key=lambda row: (row["generation"], row["decision_id"]))
        for index, row in enumerate(rows):
            previous = rows[index - 1]["decision_id"] if index else None
            following = rows[index + 1]["decision_id"] if index + 1 < len(rows) else None
            if row["previous_decision_id"] != previous or row["next_decision_id"] != following:
                errors.append(f"{row['decision_id']}: decision order mismatch")

    replay_rows = [{key: row.get(key) for key in ("condition", "ordering", "seed", "task_id", "generation", "strategy", "model_type", "metric", "best_so_far", "failure")} for row in raw]
    replay_fingerprint = _fingerprint(replay_rows)
    transfer_replay_fingerprint = _fingerprint([
        {key: row.get(key) for key in ("source_task_id", "target_task_id", "source_family", "target_family", "guard_mode", "action_taken", "transfer_score", "evidence_volume")}
        for row in transfer
    ])
    transition_rows = _read_jsonl(out / "mode_transitions.jsonl")
    routing_replay_fingerprint = _fingerprint([
        {key: row.get(key) for key in ("task_id", "state_before", "state_after", "trigger", "trigger_value", "threshold", "selected_action", "event_fingerprint")}
        for row in transition_rows
    ])
    group_keys = {(row.get("condition"), row.get("ordering"), row.get("seed"), row.get("task_id")) for row in raw}
    expected_groups = len(protocol.get("conditions", [])) * len(protocol.get("orderings", [])) * len(protocol.get("seeds", [])) * len(protocol.get("task_universe", []))
    storage_bytes = sum(path.stat().st_size for path in out.iterdir() if path.is_file())
    manifest["replay_fingerprint"] = replay_fingerprint
    passed = (
        not errors
        and bool(raw)
        and protocol.get("status") == "FROZEN_PILOT_PROTOCOL"
        and len(group_keys) == expected_groups
    )
    return {
        "validator_version": "13.8.0",
        "status": "PASS" if passed else "FAIL",
        "trajectory_count": len(ordered_groups),
        "record_count": len(raw),
        "state_count": len(states),
        "rdg_node_count": len(rdg),
        "transfer_decision_count": len(transfer),
        "semantic_replay_fingerprint": replay_fingerprint,
        "transfer_replay_fingerprint": transfer_replay_fingerprint,
        "routing_replay_fingerprint": routing_replay_fingerprint,
        "condition_seed_isolation": len(group_keys) == expected_groups,
        "expected_trajectory_groups": expected_groups,
        "observed_trajectory_groups": len(group_keys),
        "storage_bytes": storage_bytes,
        "runtime_seconds": manifest.get("elapsed_seconds"),
        "protocol_fingerprint": protocol.get("fingerprint"),
        "artifact_manifest_fingerprint": manifest.get("fingerprint"),
        "errors": errors,
    }
