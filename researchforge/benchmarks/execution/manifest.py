"""researchforge/benchmarks/execution/manifest.py — Execution Manifest Generator.

Constructs the canonical 540-entry RunManifest for Phase 12B:
- 6 experimental conditions
- 3 task permutations (forward, reverse, cross_first)
- 5 seeds [0, 1, 2, 3, 4]
- 6 tasks in the frozen cohort universe
- 90 sequence groups, 540 task evaluations, 32,400 planned candidate trials.
- Explicit sequence_transfer_regime vs. condition_effective_regime separation.
- Causal DAG dependency tracking per sequence group.
"""
from __future__ import annotations

import platform
import sys
from typing import Dict, List, Optional, Tuple

from ..cohort.models import (
    BenchmarkCohortSpecification,
    BenchmarkCondition,
    TaskGenerationSpec,
    TransferRegime,
    determine_sequence_transfer_regime,
)
from ..cohort.specification import create_canonical_cohort_spec
from ..statistical.plan import StatisticalAnalysisPlan, create_canonical_sap
from .models import ExecutionManifest, RunManifestEntry, get_software_commit_info


def get_task_sequence_for_ordering(
    tasks: Dict[str, TaskGenerationSpec], ordering: str
) -> List[TaskGenerationSpec]:
    """Returns the frozen 6-task sequence permuted according to ordering identifier."""
    base_order = [
        tasks["digits_0_4"],
        tasks["digits_5_9"],
        tasks["digits_all_10"],
        tasks["synthetic_ecg_lead1"],
        tasks["synthetic_ecg_lead2"],
        tasks["xor_parity_8bit"],
    ]

    if ordering == "forward":
        return list(base_order)
    elif ordering == "reverse":
        return list(reversed(base_order))
    elif ordering == "cross_first":
        # Cross-family task first: synthetic_ecg_lead1 placed at index 0, followed by digits and remaining
        ecg_lead1 = tasks["synthetic_ecg_lead1"]
        rest = [t for t in base_order if t.task_id != "synthetic_ecg_lead1"]
        return [ecg_lead1] + rest
    else:
        raise ValueError(f"Unknown ordering '{ordering}'. Must be 'forward', 'reverse', or 'cross_first'.")


def compute_environment_fingerprint() -> str:
    """Fingerprints system runtime environment."""
    env_str = f"{platform.system()}|{platform.release()}|{platform.machine()}|{sys.version.split()[0]}"
    import hashlib
    return hashlib.sha256(env_str.encode("utf-8")).hexdigest()


def generate_execution_manifest(
    cohort_spec: Optional[BenchmarkCohortSpecification] = None,
    sap: Optional[StatisticalAnalysisPlan] = None,
) -> ExecutionManifest:
    """Generates the sealed, complete 540-entry ExecutionManifest."""
    cohort = cohort_spec or create_canonical_cohort_spec()
    sap_plan = sap or create_canonical_sap()

    conditions = [
        BenchmarkCondition.COLD_START,
        BenchmarkCondition.NO_MEMORY,
        BenchmarkCondition.FLAT_ECRM,
        BenchmarkCondition.TRAJECTORY_MEMORY,
        BenchmarkCondition.ADAPTIVE_TRAJECTORY,
        BenchmarkCondition.CONTINUOUS_EXPERIENCE,
    ]

    orderings = list(cohort.orderings)  # ["forward", "reverse", "cross_first"]
    seeds = list(cohort.seeds)          # [0, 1, 2, 3, 4]
    tasks = cohort.tasks

    manifest_entries: List[RunManifestEntry] = []
    seen_entry_ids: set[str] = set()
    counter = 1

    for cond in conditions:
        for ord_name in orderings:
            ordered_tasks = get_task_sequence_for_ordering(tasks, ord_name)

            for seed in seeds:
                seq_group_id = f"{cond.value}__{ord_name}__seed_{seed}"
                group_entry_ids: List[str] = []

                for pos, task_spec in enumerate(ordered_tasks):
                    entry_id = f"run_{counter:04d}_{cond.value}__{ord_name}__s{seed}__p{pos}_{task_spec.task_id}"

                    prior_ids = tuple(t.task_id for t in ordered_tasks[:pos])
                    prior_fams = tuple(t.task_family for t in ordered_tasks[:pos])

                    # 1. Structural sequence transfer regime
                    seq_regime = determine_sequence_transfer_regime(prior_fams, task_spec.task_family)

                    # 2. Condition-effective regime:
                    # In COLD_START and NO_MEMORY, memory does not persist across tasks.
                    # Therefore, even if the sequence is SAME_FAMILY, the agent's effective experience is NO_PRIOR_EXPERIENCE.
                    if cond in (BenchmarkCondition.COLD_START, BenchmarkCondition.NO_MEMORY):
                        eff_regime = TransferRegime.NO_PRIOR_EXPERIENCE
                    else:
                        eff_regime = seq_regime

                    # DAG dependencies: strictly preceding entries in the same sequence group
                    depends_on = tuple(group_entry_ids)

                    entry = RunManifestEntry(
                        entry_id=entry_id,
                        condition=cond,
                        seed=seed,
                        ordering=ord_name,
                        sequence_group_id=seq_group_id,
                        sequence_position=pos,
                        task_id=task_spec.task_id,
                        task_family=task_spec.task_family,
                        task_fingerprint=task_spec.task_fingerprint,
                        prior_task_ids=prior_ids,
                        prior_task_families=prior_fams,
                        sequence_transfer_regime=seq_regime,
                        condition_effective_regime=eff_regime,
                        depends_on_entry_ids=depends_on,
                        generation_budget=10,
                        population_size=6,
                        timeout_seconds=30.0,
                        cohort_fingerprint=cohort.cohort_fingerprint,
                        sap_fingerprint=sap_plan.plan_fingerprint,
                    )
                    entry_fp = entry.compute_fingerprint()
                    entry = RunManifestEntry(**{**entry.__dict__, "entry_fingerprint": entry_fp})

                    assert entry_id not in seen_entry_ids, f"Duplicate entry_id detected: {entry_id}"
                    seen_entry_ids.add(entry_id)
                    group_entry_ids.append(entry_id)
                    manifest_entries.append(entry)
                    counter += 1

    git_sha, _ = get_software_commit_info()
    manifest = ExecutionManifest(
        manifest_version="1.0.0",
        cohort_version=cohort.specification_version,
        cohort_fingerprint=cohort.cohort_fingerprint,
        sap_version=sap_plan.plan_version,
        sap_fingerprint=getattr(sap_plan, "plan_fingerprint", getattr(sap_plan, "sap_fingerprint", "")),
        researchforge_release="1.0.0-alpha.3",
        git_commit=git_sha,
        environment_fingerprint=compute_environment_fingerprint(),
        entry_count=len(manifest_entries),
        planned_trials=len(manifest_entries) * 60,
        sequence_group_count=len(conditions) * len(orderings) * len(seeds),
        entries=tuple(manifest_entries),
    )

    manifest_fp = manifest.compute_fingerprint()
    return ExecutionManifest(**{**manifest.__dict__, "manifest_fingerprint": manifest_fp})
